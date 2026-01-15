"""
Colored Print Utilities for WorldMind Module
"""

import re


class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    BOLD_BLACK = '\033[1;30m'
    BOLD_RED = '\033[1;31m'
    BOLD_GREEN = '\033[1;32m'
    BOLD_YELLOW = '\033[1;33m'
    BOLD_BLUE = '\033[1;34m'
    BOLD_MAGENTA = '\033[1;35m'
    BOLD_CYAN = '\033[1;36m'
    BOLD_WHITE = '\033[1;37m'
    
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


MODULE_COLORS = {
    'agent': Colors.BOLD_CYAN,
    'summarizer': Colors.BOLD_GREEN,
    'discriminator': Colors.BOLD_YELLOW,
    'reflector': Colors.BOLD_MAGENTA,
    'trajectory': Colors.BOLD_BLUE,
}


def _get_color(module: str) -> str:
    """Get the color for a module."""
    return MODULE_COLORS.get(module, Colors.WHITE)


def _extract_user_content_from_prompt(prompt: str) -> str:
    """Extract user-specific content from prompt."""
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


def print_agent_input(prompt: str, detailed_output: bool = True):
    """Print agent input in cyan."""
    if not detailed_output:
        return
    
    color = _get_color('agent')
    print(f"{color}[AGENT INPUT]{Colors.RESET}")
    
    user_content = _extract_user_content_from_prompt(prompt)
    
    if len(user_content) > 3000:
        print(f"{color}  {user_content[:6000]}...{Colors.RESET}")
    else:
        print(f"{color}  {user_content}{Colors.RESET}")
    print()


def print_agent(message: str, detailed_output: bool = True):
    """Print agent output in cyan."""
    if not detailed_output:
        return
    color = _get_color('agent')
    print(f"{color}[AGENT OUTPUT]{Colors.RESET}")
    print(f"{color}{message}{Colors.RESET}")
    print()


def print_summarizer_input(before_state: str, after_state: str, action: str, detailed_output: bool = True):
    """Print summarizer input in green."""
    if not detailed_output:
        return
    color = _get_color('summarizer')
    print(f"{color}[SUMMARIZER INPUT]{Colors.RESET}")
    print(f"{color}  Action: {action}{Colors.RESET}")
    print(f"{color}  (Processing before/after images...){Colors.RESET}")
    print()


def print_summarizer_output(state_before: str, state_after: str, detailed_output: bool = True):
    """Print summarizer output in green."""
    if not detailed_output:
        return
    color = _get_color('summarizer')
    print(f"{color}[SUMMARIZER OUTPUT]{Colors.RESET}")
    print(f"{color}  State Before Action: {state_before}{Colors.RESET}")
    print(f"{color}  State After Action: {state_after}{Colors.RESET}")
    print()


def print_discriminator_input(
    predicted_state: str, 
    actual_state_summary: str, 
    action: str, 
    detailed_output: bool = True
):
    """Print discriminator input in yellow."""
    if not detailed_output:
        return
    color = _get_color('discriminator')
    print(f"{color}[DISCRIMINATOR INPUT]{Colors.RESET}")
    print(f"{color}  Action: {action}{Colors.RESET}")
    print(f"{color}  Predicted State: {predicted_state}{Colors.RESET}")
    print(f"{color}  Actual State Summary: {actual_state_summary}{Colors.RESET}")
    print()


def print_discriminator_output(match: bool, reason: str, detailed_output: bool = True):
    """Print discriminator output in yellow."""
    if not detailed_output:
        return
    color = _get_color('discriminator')
    match_str = "MATCH" if match else "MISMATCH"
    match_color = Colors.GREEN if match else Colors.RED
    print(f"{color}[DISCRIMINATOR OUTPUT]{Colors.RESET}")
    print(f"{color}  Match: {match_color}{match_str}{Colors.RESET}")
    print(f"{color}  Reason: {reason}{Colors.RESET}")
    print()


def print_reflector_input(
    action: str,
    predicted_state: str,
    state_before: str,
    state_after: str,
    discrimination_reason: str,
    detailed_output: bool = True
):
    """Print reflector input variables in magenta."""
    if not detailed_output:
        return
    color = _get_color('reflector')
    print(f"{color}[REFLECTOR INPUT VARIABLES]{Colors.RESET}")
    print(f"{color}  Action: {action}{Colors.RESET}")
    print(f"{color}  Predicted State: {predicted_state}{Colors.RESET}")
    print(f"{color}  State Before Action: {state_before}{Colors.RESET}")
    print(f"{color}  State After Action: {state_after}{Colors.RESET}")
    print(f"{color}  Discrimination Reason: {discrimination_reason}{Colors.RESET}")
    print()


def print_reflector_output(reflexion: str, detailed_output: bool = True):
    """Print reflector output in magenta."""
    if not detailed_output:
        return
    color = _get_color('reflector')
    print(f"{color}[REFLECTOR OUTPUT]{Colors.RESET}")
    print(f"{color}  Reflexion: {reflexion}{Colors.RESET}")
    print()


def print_trajectory_entry(
    step: int,
    observation: str,
    action: str,
    predicted_state: str,
    reflexion: str = None,
    next_observation: str = None,
    detailed_output: bool = True
):
    """Print experience trajectory entry in blue."""
    if not detailed_output:
        return
    color = _get_color('trajectory')
    print(f"{color}[TRAJECTORY ENTRY - Step {step}]{Colors.RESET}")
    print(f"{color}  Observation: {observation[:100]}...{Colors.RESET}")
    print(f"{color}  Action: {action}{Colors.RESET}")
    print(f"{color}  Predicted State: {predicted_state}{Colors.RESET}")
    if reflexion:
        print(f"{color}  Reflexion: {reflexion}{Colors.RESET}")
    if next_observation:
        print(f"{color}  Next Observation: {next_observation[:100]}...{Colors.RESET}")
    print()


def print_trajectory_retrieval(
    current_action: str,
    retrieved_entries: list,
    detailed_output: bool = True
):
    """Print trajectory retrieval results in blue."""
    if not detailed_output:
        return
    color = _get_color('trajectory')
    print(f"{color}[TRAJECTORY RETRIEVAL]{Colors.RESET}")
    print(f"{color}  Current Action: {current_action}{Colors.RESET}")
    print(f"{color}  Retrieved {len(retrieved_entries)} similar experiences:{Colors.RESET}")
    for i, entry in enumerate(retrieved_entries):
        print(f"{color}    {i+1}. Action: {entry.get('action', 'N/A')} (similarity: {entry.get('similarity', 'N/A'):.3f}){Colors.RESET}")
    print()


def print_separator(module: str = 'agent', detailed_output: bool = True):
    """Print a separator line."""
    if not detailed_output:
        return
    color = _get_color(module)
    print(f"{color}{'=' * 60}{Colors.RESET}")
