import os
import av
import torch
import folder_paths
import json
import math
import numpy as np
from typing import Optional
from typing_extensions import override
from fractions import Fraction
from comfy_api.latest import ComfyExtension, io, ui, Input, InputImpl, Types
from comfy.cli_args import args

class SaveWEBM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveWEBM",
            search_aliases=["export webm"],
            display_name="Save WEBM",
            category="video",
            is_experimental=True,
            inputs=[
                io.Image.Input("images", tooltip="RGBA images are saved with their alpha channel as transparency (vp9 codec only)."),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.Combo.Input("codec", options=["vp9", "av1"]),
                io.Float.Input("fps", default=24.0, min=0.01, max=1000.0, step=0.01),
                io.Float.Input("crf", default=32.0, min=0, max=63.0, step=1, tooltip="Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, codec, fps, filename_prefix, crf) -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )

        file = f"{filename}_{counter:05}_.webm"
        container = av.open(os.path.join(full_output_folder, file), mode="w")

        if cls.hidden.prompt is not None:
            container.metadata["prompt"] = json.dumps(cls.hidden.prompt)

        if cls.hidden.extra_pnginfo is not None:
            for x in cls.hidden.extra_pnginfo:
                container.metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        # Save transparency when the images carry an alpha channel (RGBA) and the codec supports it.
        # vp9 -> yuva420p; other codecs have no usable alpha path, so the alpha is ignored.
        save_alpha = images.shape[-1] == 4 and codec == "vp9"

        codec_map = {"vp9": "libvpx-vp9", "av1": "libsvtav1"}
        stream = container.add_stream(codec_map[codec], rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[-2]
        stream.height = images.shape[-3]
        stream.pix_fmt = "yuva420p" if save_alpha else ("yuv420p10le" if codec == "av1" else "yuv420p")
        stream.bit_rate = 0
        stream.options = {'crf': str(crf)}
        if codec == "av1":
            stream.options["preset"] = "6"

        for frame in images:
            if save_alpha:
                frame = av.VideoFrame.from_ndarray(torch.clamp(frame[..., :4] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy(), format="rgba")
            else:
                frame = av.VideoFrame.from_ndarray(torch.clamp(frame[..., :3] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy(), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        container.mux(stream.encode())
        container.close()

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))

class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideo",
            search_aliases=["export video"],
            display_name="Save Video",
            category="video",
            essentials_category="Basics",
            description="Saves the input images to your ComfyUI output directory.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto", tooltip="The codec to use for the video."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix, format: str, codec) -> io.NodeOutput:
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=saved_metadata
        )

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


class AccumulateSaveVideo(io.ComfyNode):
    _states = {}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AccumulateSaveVideo",
            search_aliases=["accumulate video", "export accumulated video"],
            display_name="Accumulate Save Video",
            category="video",
            description="Encodes video chunks across executions and saves when last is true.",
            inputs=[
                io.Video.Input("video", tooltip="The video chunk to append."),
                io.String.Input("filename_prefix", default="video/ComfyUI"),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto"),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto"),
                io.Boolean.Input("last", default=False),
                io.Audio.Input("complete_audio", optional=True, tooltip="Optional complete audio track to write once at finalization."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix, format: str, codec, last: bool, complete_audio: Optional[Input.Audio] = None) -> io.NodeOutput:
        node_id = cls.hidden.unique_id or "default"
        components = video.get_components()
        state = cls._states.get(node_id)
        if state is None:
            if Types.VideoContainer(format) not in (Types.VideoContainer.AUTO, Types.VideoContainer.MP4):
                raise ValueError("Only MP4 format is supported for now")
            if Types.VideoCodec(codec) not in (Types.VideoCodec.AUTO, Types.VideoCodec.H264):
                raise ValueError("Only H264 codec is supported for now")

            width, height = video.get_dimensions()
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(), width, height
            )
            file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
            path = os.path.join(full_output_folder, file)
            metadata = None
            if not args.disable_metadata:
                metadata = {}
                if cls.hidden.extra_pnginfo is not None:
                    metadata.update(cls.hidden.extra_pnginfo)
                if cls.hidden.prompt is not None:
                    metadata["prompt"] = cls.hidden.prompt
                if len(metadata) == 0:
                    metadata = None

            extra_kwargs = {"format": Types.VideoContainer(format).value} if Types.VideoContainer(format) != Types.VideoContainer.AUTO else {}
            output = av.open(path, mode="w", options={"movflags": "use_metadata_tags"}, **extra_kwargs)
            if metadata is not None:
                for key, value in metadata.items():
                    output.metadata[key] = json.dumps(value)

            frame_rate = Fraction(round(components.frame_rate * 1000), 1000)
            is_10bit = video.get_bit_depth() >= 10
            pix_fmt = "yuv420p10le" if is_10bit else "yuv420p"
            video_stream = output.add_stream("h264", rate=frame_rate)
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = pix_fmt

            audio = complete_audio or components.audio
            audio_stream = None
            audio_sample_rate = 1
            if audio:
                audio_sample_rate = int(audio["sample_rate"])
                channels = audio["waveform"].shape[1]
                layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")
                audio_stream = output.add_stream("aac", rate=audio_sample_rate, layout=layout)

            state = cls._states[node_id] = {
                "path": path, "file": file, "subfolder": subfolder, "output": output,
                "video_stream": video_stream, "audio_stream": audio_stream, "audio_sample_rate": audio_sample_rate,
                "frame_rate": frame_rate, "frame_count": 0, "is_10bit": is_10bit, "pix_fmt": pix_fmt,
                "complete_audio": complete_audio, "audio_chunks": [],
            }

        try:
            for frame in components.images:
                if state["is_10bit"]:
                    img = (frame.float() * 65535).clamp(0, 65535).cpu().numpy().astype(np.uint16)
                    frame = av.VideoFrame.from_ndarray(img, format="rgb48le")
                else:
                    img = (frame * 255).clamp(0, 255).byte().cpu().numpy()
                    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame = frame.reformat(format=state["pix_fmt"])
                for packet in state["video_stream"].encode(frame):
                    state["output"].mux(packet)
                state["frame_count"] += 1

            if state["complete_audio"] is None and complete_audio is not None:
                state["complete_audio"] = complete_audio
            if state["complete_audio"] is None and components.audio:
                state["audio_chunks"].append(components.audio["waveform"][0])

            if not last:
                return io.NodeOutput()

            for packet in state["video_stream"].encode(None):
                state["output"].mux(packet)
            audio = complete_audio or state["complete_audio"]
            if audio is None and state["audio_chunks"]:
                audio = {"sample_rate": state["audio_sample_rate"], "waveform": torch.cat(state["audio_chunks"], dim=1).unsqueeze(0)}
            if state["audio_stream"] and audio:
                waveform = audio["waveform"][0, :, :math.ceil((state["audio_sample_rate"] / state["frame_rate"]) * state["frame_count"])]
                layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(waveform.shape[0], "stereo")
                frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format="fltp", layout=layout)
                frame.sample_rate = state["audio_sample_rate"]
                frame.pts = 0
                for packet in state["audio_stream"].encode(frame):
                    state["output"].mux(packet)
                for packet in state["audio_stream"].encode(None):
                    state["output"].mux(packet)
            state["output"].close()
            cls._states.pop(node_id, None)
            return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(state["file"], state["subfolder"], io.FolderType.output)]))
        except Exception:
            state["output"].close()
            cls._states.pop(node_id, None)
            if os.path.exists(state["path"]):
                os.remove(state["path"])
            raise


class CreateVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateVideo",
            search_aliases=["images to video"],
            display_name="Create Video",
            category="video",
            essentials_category="Video Tools",
            description="Create a video from images.",
            inputs=[
                io.Image.Input("images", tooltip="The images to create a video from."),
                io.Float.Input("fps", default=30.0, min=1.0, max=120.0, step=1.0),
                io.Audio.Input("audio", optional=True, tooltip="The audio to add to the video."),
                io.Int.Input(
                    "bit_depth",
                    min=8,
                    max=10,
                    default=8,
                    step=2,
                    tooltip="Bit depth of the created video. 10-bit keeps smoother gradients with less"
                    " banding, but some players and downstream nodes may not support it.",
                    optional=True,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(
        cls, images: Input.Image, fps: float, audio: Optional[Input.Audio] = None, bit_depth: int = 8,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            InputImpl.VideoFromComponents(
                Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)),
                bit_depth=bit_depth,
            )
        )

class GetVideoComponents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetVideoComponents",
            search_aliases=["extract frames", "split video", "video to images", "demux"],
            display_name="Get Video Components",
            category="video",
            description="Extracts all components from a video: frames, audio, framerate, and bit depth.",
            inputs=[
                io.Video.Input("video", tooltip="The video to extract components from."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="bit_depth"),
            ],
        )

    @classmethod
    def execute(cls, video: Input.Video) -> io.NodeOutput:
        components = video.get_components()
        return io.NodeOutput(components.images, components.audio, float(components.frame_rate), video.get_bit_depth())


class LoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="LoadVideo",
            search_aliases=["import video", "open video", "video file"],
            display_name="Load Video",
            category="video",
            essentials_category="Basics",
            inputs=[
                io.Combo.Input("file", options=sorted(files), upload=io.UploadType.video),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        video_path = folder_paths.get_annotated_filepath(file)
        return io.NodeOutput(InputImpl.VideoFromFile(video_path))

    @classmethod
    def fingerprint_inputs(s, file):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def validate_inputs(s, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)

        return True

class VideoSlice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Video Slice",
            display_name="Video Slice",
            search_aliases=[
                "trim video duration",
                "skip first frames",
                "frame load cap",
                "start time",
            ],
            category="video",
            essentials_category="Video Tools",
            inputs=[
                io.Video.Input("video"),
                io.Float.Input(
                    "start_time",
                    default=0.0,
                    max=1e5,
                    min=-1e5,
                    step=0.001,
                    tooltip="Start time in seconds",
                ),
                io.Float.Input(
                    "duration",
                    default=0.0,
                    min=0.0,
                    step=0.001,
                    tooltip="Duration in seconds, or 0 for unlimited duration",
                ),
                io.Boolean.Input(
                    "strict_duration",
                    default=False,
                    tooltip="If True, when the specified duration is not possible, an error will be raised.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, video: io.Video.Type, start_time: float, duration: float, strict_duration: bool) -> io.NodeOutput:
        trimmed = video.as_trimmed(start_time, duration, strict_duration=strict_duration)
        if trimmed is not None:
            return io.NodeOutput(trimmed)
        raise ValueError(
            f"Failed to slice video:\nSource duration: {video.get_duration()}\nStart time: {start_time}\nTarget duration: {duration}"
        )


class VideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SaveWEBM,
            SaveVideo,
            AccumulateSaveVideo,
            CreateVideo,
            GetVideoComponents,
            LoadVideo,
            VideoSlice,
        ]

async def comfy_entrypoint() -> VideoExtension:
    return VideoExtension()
