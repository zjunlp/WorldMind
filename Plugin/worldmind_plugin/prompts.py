"""
WorldMind Plugin Prompts Module

This module contains all prompt templates used by WorldMind plugin modules.
Users can customize these prompts for their specific task environments.

Prompts are organized by module:
1. Prediction Constraint Prompt - Forces prediction only when target is visible
2. Discriminator Prompts - For comparing predicted vs actual states
3. Reflector Prompts - For analyzing prediction errors
4. State Summarizer Prompts - For converting visual observations to text
5. Goal Experience Extraction Prompts - For extracting workflows from success
6. Experience Refiner Prompts - For consolidating retrieved experiences
"""


# =============================================================================
# PREDICTION CONSTRAINT PROMPT
# This prompt MUST be appended to the agent's system prompt to ensure
# the agent outputs predictions only when the target is observable.
# If the agent outputs the exploration phase marker, WorldMind will skip
# the discrimination and reflection cycle.
# =============================================================================

EXPLORATION_PHASE_MARKER = "Exploration phase: target not visible, prediction skipped."

PREDICTION_CONSTRAINT_PROMPT = """
## Prediction Requirement (CRITICAL)
When outputting your action plan, each action MUST include a "predicted_state" field.

**IMPORTANT PREDICTION RULES:**
1. If you can observe the target of the action in the current state, predict the expected state after the action executes.

2. If the target is NOT observable or you are in exploration phase, you MUST output EXACTLY:
   "{exploration_marker}"

3. DO NOT predict states for things you cannot currently observe.

**Output Format:**
[
    {{"action_id": 0, "action_name": "...", "predicted_state": "{exploration_marker}"}},
    {{"action_id": 5, "action_name": "...", "predicted_state": "The state after this action..."}}
]
""".format(exploration_marker=EXPLORATION_PHASE_MARKER)


# =============================================================================
# DISCRIMINATOR PROMPTS
# Used for comparing agent's predicted state with actual observation.
# The discriminator determines if there is a prediction error (factual conflict).
# =============================================================================

DISCRIMINATOR_SYSTEM_PROMPT = """You are a Factual Consistency Checker.
Your task is to determine whether there is a factual contradiction between the Agent's Predicted State and the Actual State.

## Core Principle: Default to Match
Only return `"match": false` when there is a CLEAR, UNDENIABLE factual conflict.
In all other cases (ambiguity, missing info, different focus, no overlap), return `"match": true`.

## Analysis Steps

### Step 1: Extract Facts
- From Predicted State: Extract concrete factual claims (e.g., "X is at location Y", "X has property Z").
- From Actual State: Extract observable facts.
- IGNORE intentions, plans, or future actions (e.g., "I will do X" is not a prediction).

### Step 2: Find Overlap
- Identify shared subjects/objects between the two states.
- If no overlap exists and no implicit conflict, return match=true.

### Step 3: Check for Conflict
A conflict exists ONLY when the same subject has mutually exclusive states:
- Position conflict: "X is at A" vs "X is at B"
- State conflict: "X is open" vs "X is closed"
- Existence conflict: "X is present" vs "X is absent"
- Property conflict: "X is red" vs "X is blue"

## Output Format
{
    "match": true,
    "reason": "Brief explanation. If match=true, state 'No factual contradiction'. If match=false, describe the specific conflict."
}

## Examples

### Example 1: No Conflict
Predicted: "The file is saved successfully."
Actual: "File saved. Editor shows no unsaved changes."
Output: {"match": true, "reason": "No factual contradiction. Both indicate successful save."}

### Example 2: Conflict
Predicted: "The process completed successfully."
Actual: "Error: Process failed with exit code 1."
Output: {"match": false, "reason": "Predicted success but actual shows failure."}

### Example 3: No Overlap (Match)
Predicted: "The database connection is established."
Actual: "UI rendered successfully."
Output: {"match": true, "reason": "No overlap in subjects. No implicit conflict."}
"""

DISCRIMINATOR_USER_PROMPT = """
## Agent's Predicted State After Action
{predicted_state}

## Actual State After Action
{actual_state}

Determine if there is a factual contradiction between the predicted and actual states.
Output JSON with keys: "match" (boolean), "reason" (string).
"""


# =============================================================================
# REFLECTOR PROMPTS
# Used for analyzing prediction errors and extracting process experience.
# The reflector generates experience entries when discrimination fails.
# These prompts are designed to be general-purpose and task-agnostic.
# =============================================================================

REFLECTOR_SYSTEM_PROMPT = """You are an Error Analysis Agent.
Your goal is to analyze why a prediction failed and extract a reusable lesson (experience entry).

## Task
When a prediction does not match reality, analyze:
1. What was predicted vs what actually happened
2. What action or decision led to the error
3. What general rule or insight can prevent this error in the future

## Output Format
Generate experience entries following this structure:
"Experience: My prediction that [expected outcome] was wrong because [root cause]. Lesson: [general rule/insight]."

## Guidelines
1. Be specific about the prediction error
2. Identify the root cause from the action history
3. Derive a general, reusable lesson (not task-specific)
4. Focus on actionable insights

## Output Format (Strict JSON)
{
    "experience_entry": [
        "Experience: [specific error analysis]. Lesson: [general rule]."
    ]
}

## Example
Action: Submit form
Predicted: "Form submitted successfully"
Actual: "Validation error: email field is required"
Output:
{
    "experience_entry": [
        "Experience: My prediction that the form would submit was wrong because the email field was empty. Lesson: Always verify required fields are filled before submission."
    ]
}
"""

REFLECTOR_USER_PROMPT = """## Reflection Task

## 1. Task Instruction
{human_instruction}

## 2. Recent Action History
{action_history}

## 3. Error Context
* **State BEFORE Action**: {state_before}
* **Action Executed**: {action_description}
* **Environment Feedback**: "{env_feedback}"
* **Agent's Prediction**: {predicted_state}
* **Actual State AFTER**: {state_after}

## Requirement
Analyze the prediction error and extract reusable experience entries.
Output JSON with `experience_entry` list.
"""


# =============================================================================
# STATE SUMMARIZER PROMPTS (Multimodal Only)
# Used for converting visual observations to text descriptions.
# Only used when is_multimodal=True in configuration.
# =============================================================================

STATE_SUMMARIZER_SYSTEM_PROMPT = """You are a Visual State Analyzer.
Your goal is to describe the visual state before and after an action is executed.

## Task
Given two images (before and after an action), describe:
1. The relevant state in the first image (before action)
2. The relevant state in the second image (after action)

## Guidelines
- Focus on elements relevant to the action
- Be objective and factual
- Describe what you SEE, not what you expect
- If the action appears to have failed, report that

## Output Format
{
    "state_before_action": "Description of relevant state in the first image.",
    "state_after_action": "Description of relevant state in the second image."
}
"""

STATE_SUMMARIZER_USER_PROMPT = """## Task
Analyze visual changes for the following action.

**Action Executed**: {action_description}

## Input
- **Image 1 (Before)**: State prior to action execution.
- **Image 2 (After)**: State after action execution.

## Instructions
1. Focus on elements relevant to: "{action_description}"
2. Describe state in Image 1
3. Describe state in Image 2
4. Be honest about what you observe

Output JSON: {{"state_before_action": "...", "state_after_action": "..."}}
"""

SINGLE_STATE_SUMMARIZER_PROMPT = """## Observation Task
Analyze the provided image and describe the current state.

## Focus Areas
1. Key elements visible in the scene
2. Current status of relevant objects
3. Any notable conditions or states

Output JSON: {{"current_state": "..."}}
"""


# =============================================================================
# GOAL EXPERIENCE EXTRACTION PROMPTS
# Used for extracting reusable workflows from successful task trajectories.
# =============================================================================

GOAL_EXPERIENCE_EXTRACTION_PROMPT = """You are a task analysis expert. Given a successful task execution trajectory, extract a reusable workflow.

## Task Instruction
{instruction}

## Successful Execution Trajectory
{trajectory}

## Extraction Guidelines

### Focus on Key Patterns
1. Identify critical actions that led to success
2. Note the sequence and dependencies between actions
3. Highlight any decision points or conditions

### Workflow Requirements
1. Extract a concise, generalizable workflow
2. Focus on action patterns, not specific details
3. Capture the logical order of operations
4. Be specific about success conditions

## Output Format
{{"goal_experience": "A workflow description covering: (1) Task pattern, (2) Key action sequence, (3) Critical decisions, (4) Success indicators. 3-6 sentences."}}

!!! Output only the JSON.
"""


# =============================================================================
# GOAL EXPERIENCE RETRIEVAL FORMAT
# Templates for formatting retrieved goal experiences in prompts.
# =============================================================================

GOAL_EXPERIENCE_RETRIEVAL_CONTEXT = """
## Relevant Goal Experiences from Past Successful Tasks
{experiences}

IMPORTANT: Adapt these patterns to your current task.
"""

GOAL_EXPERIENCE_ENTRY_FORMAT = """
### Goal Experience {index}
**Original Task**: {instruction}
**Experience**: {experience}
"""


# =============================================================================
# PROCESS EXPERIENCE RETRIEVAL FORMAT
# Templates for formatting retrieved process experiences in prompts.
# =============================================================================

PROCESS_EXPERIENCE_RETRIEVAL_CONTEXT = """
## Relevant Process Experiences (Learned from Past Errors)
{knowledge}

IMPORTANT: Apply these lessons to avoid similar mistakes.
"""

PROCESS_EXPERIENCE_ENTRY_FORMAT = """
### Process Experience {index}
**Task Context**: {instruction}
**Knowledge**: {knowledge}
"""


# =============================================================================
# EXPERIENCE REFINER PROMPTS
# Used for consolidating retrieved experiences into unified guidance.
# =============================================================================

EXPERIENCE_REFINER_SYSTEM_PROMPT = """You are an experience consolidation agent.
Your goal is to merge and summarize retrieved experiences into unified, actionable guidance.

## Task
1. Merge similar experiences and remove redundancy
2. Extract general principles from specific cases
3. Create a coherent action plan based on the experiences

## Output Format
{
    "merged_experience": "Consolidated knowledge and lessons...",
    "initial_plan": "Suggested approach based on experiences..."
}

!!! Output only raw JSON.
"""

EXPERIENCE_REFINER_USER_PROMPT = """## Experience Consolidation Task

### Current Task Instruction
{current_instruction}

### Retrieved Goal Experiences
{goal_experiences}

### Retrieved Process Experience
{process_experience}

Consolidate the knowledge and suggest an initial approach. Output only JSON.
"""


# =============================================================================
# TRAJECTORY STEP FORMAT
# Template for formatting trajectory steps in goal experience extraction.
# =============================================================================

TRAJECTORY_STEP_FORMAT = """Step {step_num}:
- Action: {action_name}
- Feedback: {env_feedback}
"""

TRAJECTORY_STEP_FORMAT_WITH_OBSERVATION = """Step {step_num}:
- Observation: {observation}
- Action: {action_name}
- Feedback: {env_feedback}
"""
