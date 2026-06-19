from comfy_extras.graph_traversal import descendants


GLOBAL_VARIABLES = {}
GLOBAL_VARIABLE_PROMPT_ID = None


def prompt_variables(dynprompt):
    global GLOBAL_VARIABLE_PROMPT_ID
    prompt_id = id(dynprompt)
    if GLOBAL_VARIABLE_PROMPT_ID != prompt_id:
        GLOBAL_VARIABLES.clear()
        GLOBAL_VARIABLE_PROMPT_ID = prompt_id
    return GLOBAL_VARIABLES


class ForLoopOpen:
    def __init__(self):
        self.prompt_id = None
        self.values = []
        self.index = 0
        self.projected_nodes = set()
        self.scheduled_nodes = set()

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

    RETURN_TYPES = ("INT", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("i", "first", "last")
    FUNCTION = "open"
    CATEGORY = "looping"

    def open(self, start, end, increment, i_outer=None, dynprompt=None, execution_list=None, unique_id=None):
        if increment == 0:
            raise ValueError("ForLoopOpen increment must not be 0")

        prompt_id = id(dynprompt)
        if self.prompt_id != prompt_id:
            self.prompt_id = prompt_id
            self.values = []
            self.index = 0
            self.projected_nodes = set()
            self.scheduled_nodes = set()

        if not self.projected_nodes:
            self.values = list(range(start, end, increment))
            self.index = -1
            self.projected_nodes = descendants(dynprompt, unique_id)
            self.scheduled_nodes = self.projected_nodes.intersection(execution_list.pendingNodes)
            execution_list.project_nodes(self.projected_nodes, self.scheduled_nodes)

        self.index += 1
        if self.index >= len(self.values):
            execution_list.release_projected_nodes(self.projected_nodes)
            self.projected_nodes = set()
            self.scheduled_nodes = set()
            return {"ui": {"text": ("<complete>",)}, "result": (None, False, True)}

        execution_list.requeue_nodes(self.scheduled_nodes, self.projected_nodes)
        execution_list.defer_staged_node()
        value = self.values[self.index]
        tags = []
        if self.index == 0:
            tags.append("(first)")
        if self.index == len(self.values) - 1:
            tags.append("(last)")
        return {"ui": {"text": (" ".join([f"i: {value}"] + tags),)}, "result": (value, self.index == 0, self.index == len(self.values) - 1)}

    @classmethod
    def IS_CHANGED(cls, start, end, increment, i_outer=None, dynprompt=None, execution_list=None, unique_id=None):
        return float("NaN")

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
            "hidden": {
                "dynprompt": "DYNPROMPT",
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    OUTPUT_NODE = True
    FUNCTION = "set"
    CATEGORY = "looping"

    def set(self, name, value, dependency=None, dynprompt=None):
        prompt_variables(dynprompt)[name] = value
        return (value,)

    @classmethod
    def IS_CHANGED(cls, name, value, dependency=None, dynprompt=None):
        return float("NaN")

class GlobalVariableGet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "variable"}),
            },
            "optional": {
                "default": ("*",),
                "dependency": ("*",),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "get"
    CATEGORY = "looping"

    def get(self, name, default=None, dependency=None, dynprompt=None):
        return (prompt_variables(dynprompt).get(name, default),)

    @classmethod
    def IS_CHANGED(cls, name, default=None, dependency=None, dynprompt=None):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "ForLoopOpen": ForLoopOpen,
    "GlobalVariableSet": GlobalVariableSet,
    "GlobalVariableGet": GlobalVariableGet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForLoopOpen": "For Loop",
    "GlobalVariableSet": "Global Variable Set",
    "GlobalVariableGet": "Global Variable Get",
}
