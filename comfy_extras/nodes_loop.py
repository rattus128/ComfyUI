from comfy_execution.graph_utils import is_link


GLOBAL_VARIABLES = {}


def descendants(dynprompt, node_id):
    children = {}
    for candidate_id in dynprompt.all_node_ids():
        node = dynprompt.get_node(candidate_id)
        for value in node.get("inputs", {}).values():
            if is_link(value):
                children.setdefault(value[0], set()).add(candidate_id)

    found = set()
    stack = list(children.get(node_id, ()))
    while stack:
        child_id = stack.pop()
        if child_id in found:
            continue
        found.add(child_id)
        stack.extend(children.get(child_id, ()))
    return found


class ForLoopOpen:
    def __init__(self):
        self.values = []
        self.index = 0
        self.projected_nodes = set()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0}),
                "end": ("INT", {"default": 4}),
                "increment": ("INT", {"default": 1}),
            },
            "optional": {
                "i_outer": ("INT",),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "execution_list": "EXECUTION_LIST",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("i",)
    FUNCTION = "open"
    CATEGORY = "looping"

    def open(self, start, end, increment, i_outer=None, dynprompt=None, execution_list=None, unique_id=None):
        if increment == 0:
            raise ValueError("ForLoopOpen increment must not be 0")

        if not self.projected_nodes:
            self.values = list(range(start, end, increment))
            self.index = -1
            self.projected_nodes = descendants(dynprompt, unique_id)
            execution_list.project_nodes(self.projected_nodes)

        self.index += 1
        if self.index >= len(self.values):
            execution_list.release_projected_nodes(self.projected_nodes)
            self.projected_nodes = set()
            return (None,)

        execution_list.requeue_nodes(self.projected_nodes)
        execution_list.defer_staged_node()
        return (self.values[self.index],)


class GlobalVariableSet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "variable"}),
                "value": ("*",),
            },
            "optional": {
                "dependency": ("*",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "set"
    CATEGORY = "looping"

    def set(self, name, value, dependency=None):
        GLOBAL_VARIABLES[name] = value
        return ()


class GlobalVariableGet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "variable"}),
            },
            "optional": {
                "dependency": ("*",),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "get"
    CATEGORY = "looping"

    def get(self, name, dependency=None):
        return (GLOBAL_VARIABLES.get(name),)

    @classmethod
    def IS_CHANGED(cls, name, dependency=None):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "ForLoopOpen": ForLoopOpen,
    "GlobalVariableSet": GlobalVariableSet,
    "GlobalVariableGet": GlobalVariableGet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForLoopOpen": "For Loop Open",
    "GlobalVariableSet": "Global Variable Set",
    "GlobalVariableGet": "Global Variable Get",
}
