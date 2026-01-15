"""
WorldMind State Prediction Discriminator for Alfred Environment

This module provides an LLM-based discriminator that compares the agent's predicted state
with the actual state summary (text) after executing an action.
Text-only mode: compares state summarizer output text with predicted state text.
"""

import os
import json
import re
from openai import OpenAI
from embodiedbench.main import logger


# Discriminator system prompt (text-based comparison) for Alfred environment
DISCRIMINATOR_SYSTEM_PROMPT = """You are a **Logical Consistency Validator** for an embodied agent in the **Alfred household environment**.
Your SOLE purpose is to detect **Factual Contradictions** between the Agent's Prediction and the Actual Observation.

## The Alfred Environment Context
- **Valid Actions**: Find, Pick up, Put down, Drop, Open, Close, Turn on, Turn off, Slice
- **Valid Receptacles**: Fridge, Cabinet, Drawer, Microwave, CounterTop, DiningTable, SideTable, Sink, StoveBurner, GarbageCan, Shelf
- **Valid Objects**: Apple, Bread, Tomato, Lettuce, Potato, Egg, Knife, Fork, Spoon, Cup, Mug, Bowl, Plate, Pot, Pan, etc.

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
    * Check for Implicit Conflict (e.g., Pred: "Holding Apple" vs Actual: "Hand Empty").
    * If no implicit conflict exists, the result is **TRUE** (Irrelevant descriptions match by default).

### Step 3: Conflict Verification
* For the Shared Objects, check for **Mutually Exclusive States**:
    * Position: "On CounterTop" vs "In Fridge" -> **FALSE**.
    * State: "Open" vs "Closed" -> **FALSE**.
    * Hand: "Holding object" vs "Hand empty" -> **FALSE**.
    * Object State: "Sliced" vs "Whole", "Cooked" vs "Raw" -> **FALSE**.
* Any difference in phrasing, detail level, or synonym usage is **NOT** a conflict.

## Output Format
Output ONLY a JSON object:
{
    "match": true, // or false
    "reason": "Explain ONLY the factual conflict if false. If true, state 'No factual contradictions found' or 'Descriptions are unrelated'."
}

## Examples

### Example 1 (Intent Filtering)
**Agent's Predicted State After Action**: "I am at the counter. I will pick up the knife to slice the tomato."
**Actual State After Action**: "The robot is facing the counter. A knife is on the counter. Hand is empty."
**Analysis**:
1. **Filter**: 
   - Prediction Fact: "At counter". (Ignore "will pick up knife" -> Intent).
   - Actual Fact: "Facing counter", "Knife on counter", "Hand empty".
2. **Intersection**: "Counter".
3. **Check**: Locations match. The "knife" intent is ignored.
**Result**:
{
    "match": true,
    "reason": "Location matches. Ignored the 'knife' in prediction as it was part of an intent statement ('will pick up'), not a state assertion."
}

### Example 2 (Factual Conflict - Hand State)
**Agent's Predicted State After Action**: "I am holding the apple."
**Actual State After Action**: "The robot's hand is empty. The apple is on the table."
**Analysis**:
1. **Filter**: Facts extracted.
2. **Intersection**: Apple, Hand.
3. **Conflict**: Pred says "Holding apple", Actual says "Hand empty".
**Result**:
{
    "match": false,
    "reason": "Factual contradiction detected: Agent predicted holding the apple, but the actual state shows the hand is empty."
}

### Example 3 (Irrelevant Descriptions - Default True)
**Agent's Predicted State After Action**: "The fridge door is closed."
**Actual State After Action**: "The robot is holding a knife. The counter is visible."
**Analysis**:
1. **Filter**: Facts extracted.
2. **Intersection**: None. (Prediction talks about Fridge; Actual talks about Knife/Counter).
3. **Conflict Check**: Does "Holding knife" contradict "Fridge closed"? No.
**Result**:
{
    "match": true,
    "reason": "No shared objects and no logical contradiction between descriptions. Defaulting to true."
}
"""


class WorldMindDiscriminator:
    """
    LLM-based discriminator for comparing predicted states with actual states.
    Text-only mode: compares predicted state text with state summarizer output text.
    Outputs match (bool) and reason (str).
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the discriminator.
        
        Args:
            model_name: The name of the LLM model to use. If None, reads from environment.
        """
        self.model_name = model_name or os.environ.get('WorldMind_DISCRIMINATOR_MODEL')
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', None)
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for WorldMind Discriminator")
        
        # Initialize OpenAI client
        if self.api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.system_prompt = DISCRIMINATOR_SYSTEM_PROMPT
        
        # Statistics tracking
        self.total_judgments = 0
        self.match_count = 0
        self.mismatch_count = 0
        
        logger.info(f"WorldMind Discriminator initialized with model: {self.model_name}, mode: text-only")
    
    def discriminate(
        self, 
        predicted_state: str, 
        actual_state_summary: str,
        action_description: str = ""
    ) -> dict:
        """
        Compare the predicted state with the actual state summary (text-based).
        
        Args:
            predicted_state: The agent's predicted state after the action (text description)
            actual_state_summary: The actual state summary text from state summarizer
            action_description: Description of the action that was executed
            
        Returns:
            dict: {"match": bool, "reason": str}
        """
        try:
            return self._discriminate_text(predicted_state, actual_state_summary, action_description)
        except Exception as e:
            logger.error(f"WorldMind Discriminator error: {e}")
            # Return default result on error, default to match to avoid blocking the process
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
        """Text mode discrimination: compare two text descriptions."""
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
        
        logger.debug(f"WorldMind Discriminator (text mode) result: match={result['match']}, reason={result['reason'][:100]}...")
        
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
        
        # Fallback parsing: determine from text
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


# Factory function
def create_discriminator(model_name: str = None) -> WorldMindDiscriminator:
    """
    Factory function to create a WorldMind Discriminator.
    
    Args:
        model_name: Model name for discrimination
        
    Returns:
        WorldMindDiscriminator instance
    """
    return WorldMindDiscriminator(model_name=model_name)
