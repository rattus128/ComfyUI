import asyncio
import copy
import json
import time
import uuid
from io import BytesIO
from urllib.parse import urlencode

import aiohttp
import torch

from comfy_api_nodes.util import ApiEndpoint, sync_op_raw
from comfy_api_nodes.util.client import _display_text
from comfy_api_nodes.util._helpers import get_comfy_api_headers
from comfy_api_nodes.util.conversions import bytesio_to_image_tensor, tensor_to_bytesio
from comfy_execution.graph import DeferredStagedNodeState
from comfy_execution.graph_utils import is_link
from comfy_execution.progress import get_progress_state
from comfy_execution.utils import get_executing_context
from comfy_extras.graph_traversal import ascendants
from server import PromptServer


RUN_LOCALLY = "RunLocally"
RUN_ON_API = "RunOnAPI"
CLOUD_BASE_URL = "https://cloud.comfy.org"
REMOTE_OUTPUT_PREFIX = "ComfyAPI"
RUN_ON_API_STATES = {}


class _ComfyApiHidden:
    def __init__(self, unique_id, auth_token_comfy_org, api_key_comfy_org, comfy_usage_source):
        self.unique_id = unique_id
        self.auth_token_comfy_org = auth_token_comfy_org
        self.api_key_comfy_org = api_key_comfy_org
        self.comfy_usage_source = comfy_usage_source


class _ComfyApiAuthNode:
    hidden = None


def _auth_node(unique_id, auth_token_comfy_org, api_key_comfy_org, comfy_usage_source):
    class AuthNode(_ComfyApiAuthNode):
        pass

    AuthNode.hidden = _ComfyApiHidden(unique_id, auth_token_comfy_org, api_key_comfy_org, comfy_usage_source)
    return AuthNode


def _class_type(dynprompt, node_id):
    return dynprompt.get_node(node_id)["class_type"]


def _remote_plan(dynprompt, image_link):
    source_id = image_link[0]
    if _class_type(dynprompt, source_id) == RUN_LOCALLY:
        raise ValueError("Run on API needs at least one remote node before it.")

    remote_nodes = ascendants(dynprompt, source_id, stop_at=lambda node_id: _class_type(dynprompt, node_id) == RUN_LOCALLY)
    remote_nodes.add(source_id)

    boundaries = set()
    for node_id in remote_nodes:
        if _class_type(dynprompt, node_id) in {RUN_LOCALLY, RUN_ON_API}:
            raise ValueError("Run on API projections cannot contain Run Locally or another Run on API node.")
        for value in dynprompt.get_node(node_id).get("inputs", {}).values():
            if is_link(value) and _class_type(dynprompt, value[0]) == RUN_LOCALLY:
                boundaries.add((value[0], value[1]))

    return {"remote_nodes": remote_nodes, "boundaries": boundaries, "output_link": image_link}


def _get_boundary_images(execution_list, unique_id, boundaries):
    images = {}
    for node_id, socket in boundaries:
        cached = execution_list.get_cache(node_id, unique_id)
        if cached is None or cached.outputs is None or socket >= len(cached.outputs):
            raise ValueError("Run on API boundary was not available after waiting for Run Locally.")
        images[(node_id, socket)] = _image_from_cached_output(cached.outputs[socket])
    return images


def _image_from_cached_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if len(output) == 1:
        return output[0]
    return torch.cat(output, dim=0)


def _remote_node_count(plan, boundary_images):
    boundary_node_count = sum(image.shape[0] * 2 - 1 for image in boundary_images.values())
    return len(plan["remote_nodes"]) + boundary_node_count + 1


def _show_status(auth_node, status, done, total, started):
    elapsed = int(time.monotonic() - started)
    _display_text(auth_node, f"Status: {status}, {done}/{total} nodes, {elapsed}s")


async def _upload_image(auth_node, image, started, total_nodes):
    _show_status(auth_node, "Uploading", 0, total_nodes, started)
    image_io = tensor_to_bytesio(image, total_pixels=None, mime_type="image/png")
    response = await sync_op_raw(
        auth_node,
        ApiEndpoint(
            f"{CLOUD_BASE_URL}/api/upload/image",
            method="POST",
            headers=get_comfy_api_headers(auth_node),
        ),
        data={"type": "input", "overwrite": "true"},
        files={"image": (image_io.name, image_io, "image/png")},
        content_type="multipart/form-data",
        timeout=120.0,
        monitor_progress=False,
    )
    return response["name"]


async def _upload_image_batch(auth_node, image, started, total_nodes):
    return [await _upload_image(auth_node, image[i], started, total_nodes) for i in range(image.shape[0])]


def _add_remote_image_input(prompt, node_prefix, filenames):
    previous = None
    for index, filename in enumerate(filenames):
        load_id = f"{node_prefix}_load_{index}"
        prompt[load_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
        if previous is None:
            previous = [load_id, 0]
            continue
        batch_id = f"{node_prefix}_batch_{index}"
        prompt[batch_id] = {"class_type": "ImageBatch", "inputs": {"image1": previous, "image2": [load_id, 0]}}
        previous = [batch_id, 0]
    return previous


async def _build_remote_prompt(auth_node, dynprompt, plan, boundary_images, started, total_nodes):
    original_prompt = dynprompt.get_original_prompt()
    prompt = {node_id: copy.deepcopy(original_prompt[node_id]) for node_id in plan["remote_nodes"]}

    for index, (boundary, image) in enumerate(boundary_images.items()):
        filenames = await _upload_image_batch(auth_node, image, started, total_nodes)
        link = _add_remote_image_input(prompt, f"__api_input_{index}", filenames)
        for node in prompt.values():
            for input_name, value in list(node.get("inputs", {}).items()):
                if value == list(boundary):
                    node["inputs"][input_name] = link

    output_id = "__api_output"
    prompt[output_id] = {
        "class_type": "SaveImage",
        "inputs": {"images": list(plan["output_link"]), "filename_prefix": REMOTE_OUTPUT_PREFIX},
    }
    return prompt, output_id


async def _submit_remote_prompt(auth_node, prompt, output_id, started):
    _show_status(auth_node, "Submitting", 0, len(prompt), started)
    response = await sync_op_raw(
        auth_node,
        ApiEndpoint(f"{CLOUD_BASE_URL}/api/prompt", method="POST", headers=get_comfy_api_headers(auth_node)),
        data={"prompt": prompt, "partial_execution_targets": [output_id]},
        timeout=120.0,
        monitor_progress=False,
    )
    return response["prompt_id"]


async def _wait_for_remote_job(auth_node, prompt_id, remote_prompt, started):
    stop_ticker = asyncio.Event()
    total_nodes = len(remote_prompt)
    progress = {"done": set(), "current": None}
    local_context = get_executing_context()

    def is_local_node(node_id):
        return node_id in remote_prompt and not node_id.startswith("__api_")

    async def ticker():
        while not stop_ticker.is_set():
            _show_status(auth_node, "Running", len(progress["done"]), total_nodes, started)
            await asyncio.sleep(1.0)

    token = auth_node.hidden.api_key_comfy_org or auth_node.hidden.auth_token_comfy_org
    ws_url = f"wss://cloud.comfy.org/ws?{urlencode({'clientId': str(uuid.uuid4()), 'token': token})}"
    ticker_task = asyncio.create_task(ticker())
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, heartbeat=30) as ws:
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(message.data)
                    data = event.get("data", {})
                    if data.get("prompt_id") != prompt_id:
                        continue

                    message_type = event.get("type")
                    if message_type == "execution_cached":
                        progress["done"].update(data.get("nodes", []))
                    elif message_type == "executing":
                        current = data.get("node")
                        if progress["current"] and current != progress["current"]:
                            progress["done"].add(progress["current"])
                            if is_local_node(progress["current"]) and local_context is not None:
                                get_progress_state().finish_progress(progress["current"])
                        progress["current"] = current
                        if is_local_node(current) and local_context is not None:
                            get_progress_state().start_progress(current)
                            PromptServer.instance.last_node_id = current
                            PromptServer.instance.send_sync(
                                "executing",
                                {"node": current, "display_node": current, "prompt_id": local_context.prompt_id},
                                PromptServer.instance.client_id,
                            )
                    elif message_type == "executed" and data.get("node"):
                        progress["done"].add(data["node"])
                        if is_local_node(data["node"]) and local_context is not None:
                            get_progress_state().finish_progress(data["node"])
                    elif message_type == "progress" and local_context is not None:
                        current = data.get("node") or progress["current"]
                        if is_local_node(current):
                            get_progress_state().update_progress(current, data["value"], data["max"])
                            PromptServer.instance.send_sync(
                                "progress",
                                {
                                    "node": current,
                                    "prompt_id": local_context.prompt_id,
                                    "value": data["value"],
                                    "max": data["max"],
                                },
                                PromptServer.instance.client_id,
                            )
                    elif message_type == "progress_state" and local_context is not None:
                        for node_id, state in data.get("nodes", {}).items():
                            if not is_local_node(node_id):
                                continue
                            if state.get("state") == "finished":
                                get_progress_state().finish_progress(node_id)
                            else:
                                get_progress_state().update_progress(node_id, state["value"], state["max"])
                    elif message_type == "execution_success":
                        progress["done"].update(remote_prompt)
                        if is_local_node(progress["current"]) and local_context is not None:
                            get_progress_state().finish_progress(progress["current"])
                        _show_status(auth_node, "Completed", total_nodes, total_nodes, started)
                        return
                    elif message_type == "execution_error":
                        raise RuntimeError(data.get("exception_message") or "API workflow failed")
    finally:
        stop_ticker.set()
        await ticker_task


async def _download_remote_images(auth_node, prompt_id, started, total_nodes):
    _show_status(auth_node, "Downloading", total_nodes, total_nodes, started)
    job = await sync_op_raw(
        auth_node,
        ApiEndpoint(f"{CLOUD_BASE_URL}/api/jobs/{prompt_id}", headers=get_comfy_api_headers(auth_node)),
        timeout=120.0,
        monitor_progress=False,
    )
    images = []
    for output in job.get("outputs", {}).values():
        images.extend(output.get("images", []))

    tensors = []
    for image in images:
        data = await sync_op_raw(
            auth_node,
            ApiEndpoint(
                f"{CLOUD_BASE_URL}/api/view",
                query_params={"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")},
                headers=get_comfy_api_headers(auth_node),
            ),
            timeout=120.0,
            as_binary=True,
            monitor_progress=False,
            progress_origin_ts=started,
        )
        tensors.append(bytesio_to_image_tensor(BytesIO(data), mode="RGB"))

    if not tensors:
        raise RuntimeError("API workflow completed without image outputs.")
    _show_status(auth_node, "Completed", total_nodes, total_nodes, started)
    return torch.cat(tensors, dim=0)


class RunLocally:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "api"

    def run(self, image):
        return (image,)


class RunOnAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"lazy": True, "rawLink": True}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "execution_list": "EXECUTION_LIST",
                "unique_id": "UNIQUE_ID",
                "auth_token_comfy_org": "AUTH_TOKEN_COMFY_ORG",
                "api_key_comfy_org": "API_KEY_COMFY_ORG",
                "comfy_usage_source": "COMFY_USAGE_SOURCE",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "api"

    def check_lazy_status(self, image, **_kwargs):
        return []

    async def run(
        self,
        image,
        dynprompt,
        execution_list,
        unique_id,
        auth_token_comfy_org=None,
        api_key_comfy_org=None,
        comfy_usage_source=None,
    ):
        if not is_link(image):
            raise ValueError("Run on API image input must be linked.")

        state_key = (id(dynprompt), unique_id)
        state = RUN_ON_API_STATES.get(state_key)
        if state is None:
            state = _remote_plan(dynprompt, image)
            RUN_ON_API_STATES[state_key] = state
            for node_id, socket in state["boundaries"]:
                execution_list.add_strong_link(node_id, socket, unique_id)
            if state["boundaries"]:
                execution_list.defer_staged_node(state=DeferredStagedNodeState.DEFERRED)
                return (None,)

        RUN_ON_API_STATES.pop(state_key, None)
        auth_node = _auth_node(unique_id, auth_token_comfy_org, api_key_comfy_org, comfy_usage_source)
        boundary_images = _get_boundary_images(execution_list, unique_id, state["boundaries"])
        started = time.monotonic()
        total_nodes = _remote_node_count(state, boundary_images)
        remote_prompt, output_id = await _build_remote_prompt(
            auth_node, dynprompt, state, boundary_images, started, total_nodes
        )
        prompt_id = await _submit_remote_prompt(auth_node, remote_prompt, output_id, started)
        await _wait_for_remote_job(auth_node, prompt_id, remote_prompt, started)
        return (await _download_remote_images(auth_node, prompt_id, started, len(remote_prompt)),)


NODE_CLASS_MAPPINGS = {
    RUN_LOCALLY: RunLocally,
    RUN_ON_API: RunOnAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    RUN_LOCALLY: "Upload to Comfy API",
    RUN_ON_API: "Run on Comfy API",
}
