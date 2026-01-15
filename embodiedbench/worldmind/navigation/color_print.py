"""
WorldMind Navigation Color Print Utilities

Color printing utilities for debugging module inputs and outputs.
"""

import re
from typing import Optional


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _extract_user_content_from_prompt(prompt: str) -> str:
    """Extract user-specific content from prompt (after system prompt and examples)."""
    patterns = [
        r'## Now the human instruction is:',
        r'## Guidelines',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prompt)
        if match:
            return prompt[match.start():]
    
    example_pattern = r'## Task Execution Example\s+\d+:'
    matches = list(re.finditer(example_pattern, prompt))
    if matches:
        last_match = matches[-1]
        remaining = prompt[last_match.end():]
        next_section = re.search(r'\n## ', remaining)
        if next_section:
            return remaining[next_section.start():]
    
    if len(prompt) > 2000:
        return "..." + prompt[-2000:]
    return prompt


def print_separator(detailed_output: bool = True):
    """Print separator line."""
    if detailed_output:
        print(f"{Colors.CYAN}{'='*80}{Colors.ENDC}")


def print_agent_input(prompt: str, detailed_output: bool = True):
    """Print agent input (user-specific content only)."""
    if not detailed_output:
        return
    
    print(f"{Colors.BOLD}{Colors.CYAN}[AGENT INPUT]{Colors.ENDC}")
    user_content = _extract_user_content_from_prompt(prompt)
    
    if len(user_content) > 3000:
        print(f"{Colors.CYAN}  {user_content[:3000]}...{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}  {user_content}{Colors.ENDC}")
    print()


def print_agent(output: str, detailed_output: bool = True):
    """Print agent output."""
    if not detailed_output:
        return
    print(f"{Colors.BOLD}{Colors.GREEN}[AGENT OUTPUT]{Colors.ENDC}")
    print(f"{Colors.GREEN}{output}{Colors.ENDC}")
    print()


def print_summarizer_input(before_image: str, after_image: str, action: str, detailed_output: bool = True):
    """Print state summarizer input."""
    if not detailed_output:
        return
    print(f"{Colors.BOLD}{Colors.YELLOW}[SUMMARIZER INPUT]{Colors.ENDC}")
    print(f"{Colors.YELLOW}  Action: {action}{Colors.ENDC}")
    print(f"{Colors.YELLOW}  (Processing before/after images...){Colors.ENDC}")
    print()


def print_summarizer_output(state_before: str, state_after: str, detailed_output: bool = True):
    """Print state summarizer output."""
    if not detailed_output:
        return
    print(f"{Colors.BOLD}{Colors.YELLOW}[SUMMARIZER OUTPUT]{Colors.ENDC}")
    print(f"{Colors.YELLOW}  State Before Action: {state_before}{Colors.ENDC}")
    print(f"{Colors.YELLOW}  State After Action: {state_after}{Colors.ENDC}")
    print()


def print_discriminator_input(predicted: str, actual: str, action: str, detailed_output: bool = True):
    """Print discriminator input."""
    if not detailed_output:
        return
    print(f"{Colors.BOLD}{Colors.YELLOW}[DISCRIMINATOR INPUT]{Colors.ENDC}")
    print(f"{Colors.YELLOW}  Action: {action}{Colors.ENDC}")
    print(f"{Colors.YELLOW}  Predicted State: {predicted}{Colors.ENDC}")
    print(f"{Colors.YELLOW}  Actual State Summary: {actual}{Colors.ENDC}")
    print()


def print_discriminator_output(match: bool, reason: str, detailed_output: bool = True):
    """Print discriminator output."""
    if not detailed_output:
        return
    match_str = "MATCH" if match else "MISMATCH"
    match_color = Colors.GREEN if match else Colors.RED
    print(f"{Colors.BOLD}{Colors.YELLOW}[DISCRIMINATOR OUTPUT]{Colors.ENDC}")
    print(f"{Colors.YELLOW}  Match: {match_color}{match_str}{Colors.ENDC}")
    print(f"{Colors.YELLOW}  Reason: {reason}{Colors.ENDC}")
    print()


def print_reflector_input(action: str, predicted: str, state_before: str, state_after: str, reason: str, detailed_output: bool = True):
    """Print reflector input."""
    if not detailed_output:
        return
    print(f"{Colors.BOLD}{Colors.RED}[REFLECTOR INPUT VARIABLES]{Colors.ENDC}")
    print(f"{Colors.RED}  Action: {action}{Colors.ENDC}")
    print(f"{Colors.RED}  Predicted State: {predicted}{Colors.ENDC}")
    print(f"{Colors.RED}  State Before Action: {state_before}{Colors.ENDC}")
    print(f"{Colors.RED}  State After Action: {state_after}{Colors.ENDC}")
    print(f"{Colors.RED}  Discrimination Reason: {reason}{Colors.ENDC}")
    print()


def print_reflector_output(reflexion: str, detailed_output: bool = True):
    """Print reflector output."""
    if not detailed_output:
        return
    print(f"{Colors.BOLD}{Colors.RED}[REFLECTOR OUTPUT]{Colors.ENDC}")
    print(f"{Colors.RED}  Reflexion: {reflexion}{Colors.ENDC}")
    print()


def print_trajectory_entry(step: int, observation: str, action: str, predicted_state: str, 
                          reflexion: Optional[str] = None, next_observation: str = None, detailed_output: bool = True):
    """Print trajectory entry."""
    if not detailed_output:
        return
    print(f"{Colors.BOLD}{Colors.HEADER}[TRAJECTORY ENTRY - Step {step}]{Colors.ENDC}")
    print(f"{Colors.HEADER}  Observation: {observation[:100]}...{Colors.ENDC}" if len(observation) > 100 else f"{Colors.HEADER}  Observation: {observation}{Colors.ENDC}")
    print(f"{Colors.HEADER}  Action: {action}{Colors.ENDC}")
    print(f"{Colors.HEADER}  Predicted State: {predicted_state}{Colors.ENDC}")
    if reflexion:
        print(f"{Colors.HEADER}  Reflexion: {reflexion}{Colors.ENDC}")
    if next_observation:
        print(f"{Colors.HEADER}  Next Observation: {next_observation[:100]}...{Colors.ENDC}" if len(next_observation) > 100 else f"{Colors.HEADER}  Next Observation: {next_observation}{Colors.ENDC}")
    print()


def print_trajectory_retrieval(current_action: str, retrieved_entries: list, detailed_output: bool = True):
    """Print trajectory retrieval results."""
    if not detailed_output:
        return
    print(f"{Colors.BOLD}{Colors.HEADER}[TRAJECTORY RETRIEVAL]{Colors.ENDC}")
    print(f"{Colors.HEADER}  Current Action: {current_action}{Colors.ENDC}")
    print(f"{Colors.HEADER}  Retrieved {len(retrieved_entries)} similar experiences:{Colors.ENDC}")
    for i, entry in enumerate(retrieved_entries):
        print(f"{Colors.HEADER}    {i+1}. Action: {entry.get('action', 'N/A')} (similarity: {entry.get('similarity', 'N/A'):.3f}){Colors.ENDC}")
    print()
