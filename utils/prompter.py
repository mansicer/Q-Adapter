"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Mapping, Union

from utils.constants import *


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(self, sample: Mapping[str, str]) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        instruction = sample.get("instruction")
        inputs = sample.get("input", None)
        label = sample.get("label", None)
        if inputs:
            res = self.template["prompt_input"].format(instruction=instruction, input=inputs)
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class RoleBenchPrompter(object):
    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template_name = template_name
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

        self.role_profiles = json.load(open("raw/RoleBench/profiles-eng/desc.json"))
        self.role_profiles.update(json.load(open("raw/RoleBench/profiles-zh/desc.json")))
        print(f"Load {len(self.role_profiles)} role profiles")

    def generate_prompt(self, sample: Mapping[str, str]) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        role = sample.get("role")
        inputs = sample.get("input")
        label = sample.get("label", None)

        role_desc = self.role_profiles[role]
        if "zh" in self.template_name:
            instruction = ROLEBENCH_INSTRUCTION_FORMAT_ZH.format(role_name=role, role_description_and_catchphrases=role_desc)
        else:
            instruction = ROLEBENCH_INSTRUCTION_FORMAT_EN.format(role_name=role, role_description_and_catchphrases=role_desc)
        res = self.template["prompt_input"].format(instruction=instruction, input=inputs)

        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class DSPPrompter(object):
    instruction_template = "## Human:\n"
    response_template = "## Assistant:\n"

    @classmethod
    def get_instruction(cls, inputs: str):
        return cls.response_template.join(inputs.split(cls.response_template)[:-1]) + cls.response_template
    
    @classmethod
    def get_response(cls, inputs: str):
        return inputs.split(cls.response_template)[-1]


class HHRLHFPrompter(object):
    response_template = "\n\nAssistant: "

    @classmethod
    def get_instruction(cls, inputs: str):
        return cls.response_template.join(inputs.split(cls.response_template)[:-1]) + cls.response_template
    
    @classmethod
    def get_response(cls, inputs: str):
        return inputs.split(cls.response_template)[-1]
