"""
WorldMind State Prediction Discriminator for Habitat Environment

LLM-based discriminator that compares predicted state with actual state summary.
"""

import os
import json
import re
from openai import OpenAI
from embodiedbench.main import logger


DISCRIMINATOR_SYSTEM_PROMPT = """You are a **Logical Consistency Validator** for an embodied agent.
Your SOLE purpose is to detect **Factual Contradictions** between the Agent's Prediction and the Actual Observation.

## The "Default True" Principle
**CRITICAL RULE**: You must maintain a "Presumption of Innocence" for the agent.
1. **Irrelevance is True**: If the description in `Agent's Predicted State After Action` and `Actual State After Action` are unrelated (discussing different objects) and do not logically contradict each other, you MUST return `"match": true`.
2. **Key Consistency**: You are strictly comparing the value of `Agent's Predicted State After Action` against `Actual State After Action`.
3. **Conflict Driven**: **ONLY** return `false` if and only if there is a direct, undeniable factual conflict (e.g., A says "X is Open", B says "X is Closed"). In all other cases (ambiguity, missing info, different focus), return `true`.

## Your Strict Analysis Protocol (Step-by-Step)

### Step 1: Fact Extraction (The "No Intent" Filter)
Before comparing, you must clean the data.
* **Filter `Agent's Predicted State After Action`**: Extract ONLY objects and states mentioned in **Affirmative Declarative Sentences** (e.g., "I am holding X", "X is on Y").
    * **CRITICAL**: IGNORE, DISCARD, and REMOVE any objects mentioned in future tense, intent, or plan-checking sentences (e.g., "I will check for a pear", "I intend to find X"). These are thoughts, not state predictions.
* **Filter `Actual State After Action`**: Similarly, ensure you only focus on factual descriptions of what is currently visible.

### Step 2: Intersection Analysis
* Identify **Shared Objects** that exist in both the *Filtered* Prediction and the *Filtered* Actual State.
* If there are **NO** Shared Objects after filtering:
    * Check for Implicit Conflict (e.g., Pred: "Holding Apple" vs Actual: "Gripper Empty").
    * If no implicit conflict exists, the result is **TRUE** (Irrelevant descriptions match by default).

### Step 3: Conflict Verification
* For the Shared Objects, check for **Mutually Exclusive States**:
    * Position: "On Table" vs "On Sofa" -> **FALSE**.
    * State: "Open" vs "Closed" -> **FALSE**.
    * Gripper: "Full" vs "Empty" -> **FALSE**.
* Any difference in phrasing, detail level, or synonym usage is **NOT** a conflict.

## Output Format
Output ONLY a JSON object:
{
    "match": true, // or false
    "reason": "Explain ONLY the factual conflict if false. If true, state 'No factual contradictions found' or 'Descriptions are unrelated'."
}

## Examples

### Example 1 (Intent Filtering - The "Pear" Case)
**Agent's Predicted State After Action**: "I am at the left counter. I will check for a pear here to see if it is ripe."
**Actual State After Action**: "The robot is facing the left counter. The surface is empty."
**Analysis**:
1. **Filter**: 
   - Prediction Fact: "At left counter". (Ignore "check for pear" -> Intent).
   - Actual Fact: "Facing left counter", "Surface empty".
2. **Intersection**: "Left Counter".
3. **Check**: Locations match. The "pear" is ignored because it was only in an intent sentence.
**Result**:
{
    "match": true,
    "reason": "Location matches. Ignored the 'pear' in prediction as it was part of an intent statement ('will check'), not a state assertion."
}

### Example 2 (Irrelevant Descriptions - Default True)
**Agent's Predicted State After Action**: "The cabinet door is closed."
**Actual State After Action**: "The robot is holding a sponge. The floor is visible."
**Analysis**:
1. **Filter**: Facts extracted.
2. **Intersection**: None. (Prediction talks about Cabinet; Actual talks about Sponge/Floor).
3. **Conflict Check**: Does "Holding sponge" contradict "Cabinet closed"? No.
**Result**:
{
    "match": true,
    "reason": "No shared objects and no logical contradiction between descriptions. Defaulting to true."
}

### Example 3 (Factual Conflict - Explicit Mismatch)
**Agent's Predicted State After Action**: "I am holding the apple."
**Actual State After Action**: "The gripper is empty. The apple is on the table."
**Analysis**:
1. **Filter**: Facts extracted.
2. **Intersection**: Apple, Gripper.
3. **Conflict**: Pred says "Holding", Actual says "Empty".
**Result**:
{
    "match": false,
    "reason": "Factual contradiction detected: Agent predicted holding the apple, but the actual state shows the gripper is empty."
}
"""


class WorldMindDiscriminator:
    """LLM-based discriminator for comparing predicted states with actual states."""
    
    def __init__(self, model_name: str = None):
        """Initialize the discriminator."""
        self.model_name = model_name or os.environ.get('WORLDMIND_DISCRIMINATOR_MODEL')
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if self.api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.system_prompt = DISCRIMINATOR_SYSTEM_PROMPT
        
        self.total_judgments = 0
        self.match_count = 0
        self.mismatch_count = 0
        
        logger.info(f"WorldMind Discriminator initialized with model: {self.model_name}")
    
    def discriminate(
        self, 
        predicted_state: str, 
        actual_state_summary: str,
        action_description: str = ""
    ) -> dict:
        """Compare the predicted state with the actual state summary."""
        try:
            return self._discriminate_text(predicted_state, actual_state_summary, action_description)
        except Exception as e:
            logger.error(f"WorldMind Discriminator error: {e}")
            return {
                "match": True,
                "reason": f"Discriminator error: {str(e)}"
            }
    
    def _discriminate_text(
        self,
        predicted_state: str,
        actual_state_summary: str,
        action_description: str
    ) -> dict:
        """Text mode discrimination."""
        user_prompt = f"""
## Agent's Predicted State After Action
{predicted_state}

## Actual State After Action 
{actual_state_summary}

Please analyze whether the agent's predicted state matches the actual state description.
Consider the semantic meaning and whether the key aspects match.
Output your judgment in JSON format with keys: "match" (boolean), "reason" (string)."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=1024
        )
        
        result_text = response.choices[0].message.content
        result = self._parse_result(result_text)
        
        self._update_statistics(result["match"])
        
        logger.debug(f"Discriminator result: match={result['match']}")
        
        return result
    
    def _update_statistics(self, is_match: bool):
        """Update internal statistics."""
        self.total_judgments += 1
        if is_match:
            self.match_count += 1
        else:
            self.mismatch_count += 1
    
    def _parse_result(self, result_text: str) -> dict:
        """Parse the LLM output to extract the judgment result."""
        try:
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "match": bool(result.get("match", True)),
                    "reason": str(result.get("reason", "No reason provided"))
                }
        except json.JSONDecodeError:
            pass
        
        match = "true" in result_text.lower() and "match" in result_text.lower()
        return {
            "match": match,
            "reason": result_text[:500]
        }
    
    def get_statistics(self) -> dict:
        """Get discrimination statistics."""
        return {
            "total_judgments": self.total_judgments,
            "match_count": self.match_count,
            "mismatch_count": self.mismatch_count,
            "match_rate": self.match_count / self.total_judgments if self.total_judgments > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_judgments = 0
        self.match_count = 0
        self.mismatch_count = 0


def create_discriminator(model_name: str = None) -> WorldMindDiscriminator:
    """Factory function to create a WorldMind Discriminator."""
    return WorldMindDiscriminator(model_name=model_name)
