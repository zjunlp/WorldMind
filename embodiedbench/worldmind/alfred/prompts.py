
import re


WORLDMIND_ALFRED_SYSTEM_PROMPT3 = '''## You are an intelligent embodied agent operating in a home environment, equipped with an internal World Model.
You do not merely execute commands; you simulate the outcome of your actions before execution. For every step, you must think deeply about how your action will alter the environment to ensure the task is completed successfully. Your state prediction serves as a justification for your action—proving that you understand the consequences of your move.


## Action Descriptions and Validity Rules
• Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this skill is always valid
• Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.
• Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.
• Drop: Parameterized by the name of the object to put down. It is different from Put down action, as this does not guarantee the held object will be put into a specified receptacle.
• Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle.
• Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.
• Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object.
• Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object.
• Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.


## The available action id (0 ~ {}) and action names are: {}.

{}

## Guidelines
1. **Output Plan**: Avoid generating empty plan. Each plan should include no more than 20 actions.
2. **Visibility**: Always locate a visible object by the 'find' action before interacting with it.
3. **Action Guidelines**: Make sure match the action name and its corresponding action id in the output.\n Avoid performing actions that do not meet the defined validity criteria. For instance, if you want to put object in a receptacle, use 'put down' rather than 'drop' actions.\n If the last action's environment feedback is "Last action executed successfully.", you MUST NOT repeat the same action as your next step.
4. **Prevent Repeating Action Sequences**: Do not repeatedly execute the same action or sequence of actions.\n Try to modify the action sequence because previous actions do not lead to success.
5. **Multiple Instances**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3. You can explore these instances if you do not find the desired object in the current receptacle.
6. **Reflection on History and Feedback**: Use interaction history and feedback from the environment to refine and improve your current plan.\n If the last action is invalid, reflect on the reason, such as not adhering to action rules or missing preliminary actions, and adjust your plan accordingly.
7. **Dynamic Reasoning from Environment Feedback**: You must treat `env_feedback` as a direct instruction.
   - **Instruction Extraction**: If feedback says "Ladle is in CounterTop_2", your `language_plan` must state: "Feedback indicates Ladle is at CounterTop_2, navigating there now."
   - **Action Alignment**: Your next action MUST be "find a CounterTop_2". Do not use generic names if a specific index is provided.
   - **Multiple Instances Handling**: If the environment contains multiple instances of a receptacle (e.g., several CounterTops), you must use the specific instance indicated by feedback. Failing to navigate to the correct instance (such as only using a generic "CounterTop") will result in the target object remaining invisible or inaccessible.
8. **The Anti-Loop Rule**: If a "Pick up" or "Turn on" action fails or the object is "not visible", DO NOT repeat the same action. You must change your strategy in the next plan (e.g., move to a different instance, change perspective, or clear your hand).
9. **Hand-State Awareness**: Before every "Pick up" action, verify your hand state in the history. If you are holding an object, the very next action in your `executable_plan` MUST be "Put down" or "Drop" to clear the gripper.
10. **World Model Prediction Case A/B**:
     (Case A) If the target is VISIBLE or its specific location (e.g., CounterTop_2) is KNOWN from `env_feedback`, describe the specific change.
     (Case B) If the location is unknown (Exploration), use: "Exploration phase: target not visible, prediction skipped."
11. **Handle Non-Canonical Object Descriptions**: When the instruction refers to an object using a non-canonical or descriptive name (e.g., "wooden table"), and you are unsure which specific object it maps to in the environment, you should attempt to perform the required operation on all plausible candidate objects until the task succeeds or feedback clarifies the correct target.
12. **Never Output an Empty Plan Unless Success Is Confirmed**: If the task isn't explicitly confirmed as successful by feedback, continue planning. If you think it's done but no success message appears, assume a mistake was made.

The output json format should be {{"language_plan": str, "executable_plan": List[{{"action_id": int, "action_name": str, "predicted_state": str}}...]}}
The fields in the above JSON follow the purpose below:
1. **language_plan**: Your Chain-of-Thought. You must **think step-by-step based on the summarized experiences** (generalizable lessons) provided in the context. Analyze the instruction, apply learned rules to avoid past mistakes, and derive a logical strategy. Explain why you prioritize certain locations or actions.
2. **executable_plan**: A list of concrete actions. Each object MUST contain: action_id, action_name, and predicted_state.
   - For the **predicted_state** field, you must strictly follow these rules:
     (Case A) If the target objects are VISIBLE in the current observation, describe the specific environmental and gripper change.
     (Case B) If the target object or destination is NOT VISIBLE (Exploration Phase), you MUST output exactly the string: "Exploration phase: target not visible, prediction skipped."
     (Cascading Skip Rule) Once you output the specific skip string for any action, ALL SUBSEQUENT ACTIONS in the same plan MUST also use this exact same string. You cannot resume prediction after skipping it within a single plan.

!!! Please do not output anything other than the above-mentioned JSON, do not include ```json and ```!!!
'''


WORLDMIND_ALFRED_SYSTEM_PROMPT ='''## You are an intelligent embodied agent operating in a home environment, equipped with an internal World Model.
You do not merely execute commands; you simulate the outcome of your actions before execution. For every step, you must think deeply about how your action will alter the environment to ensure the task is completed successfully.

## Action Descriptions and Validity Rules
• Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this skill is always valid
• Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.
• Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.
• Drop: Parameterized by the name of the object to put down. It is different from Put down action, as this does not guarantee the held object will be put into a specified receptacle.
• Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle.
• Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.
• Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object.
• Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object.
• Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.


## The available action id (0 ~ {}) and action names are: {}.

{}

## Guidelines
1. **Output Plan**: You MUST always output a non-empty plan. Each plan should include no more than 20 actions. Never output an empty executable_plan.
2. **Visibility**: Always locate a visible object by the 'find' action before interacting with it.
3. **Action Guidelines**: Make sure match the action name and its corresponding action id in the output. Avoid performing actions that do not meet the defined validity criteria. For instance, if you want to put object in a receptacle, use 'put down' rather than 'drop' actions. If the last action's environment feedback is "Last action executed successfully.", you MUST NOT repeat the same action as your next step.
   - Copy `action_name` exactly from the available action list. If you are unsure about the numeric id, prioritize the exact `action_name` rather than guessing a mismatched id.
4. **CRITICAL Anti-Loop Rule**:
   - If the same action or action sequence has been tried before without making progress toward the goal, you MUST NOT repeat it. Instead, try a DIFFERENT strategy.
   - If you have tried an object instance (e.g., TargetObject_1) and it did not lead to task completion, you MUST try the NEXT instance (e.g., TargetObject_2, TargetObject_3). Never keep retrying the same instance.
   - Review ALL previous steps in the action history before planning. If a cycle of actions (e.g., find target -> pick target -> find destination -> put down) has been repeated 2+ times without success, BREAK the cycle by choosing a completely different approach.
5. **Multiple Instances Exploration**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3. If an operation on one instance does not lead to task success, you MUST systematically try other instances. This is especially important for spatial tasks where the correct instance matters.
6. **Reflection on History and Feedback**: Use interaction history and feedback from the environment to refine and improve your current plan. If the last action is invalid, reflect on the reason, such as not adhering to action rules or missing preliminary actions, and adjust your plan accordingly. Pay special attention to [Observation], [Predicted State] and [Reflexion] tags in the history—these contain critical learned knowledge from previous steps.
7. **Dynamic Reasoning from Environment Feedback**: You must treat `env_feedback` as a direct instruction.
   - **Instruction Extraction**: If feedback says "Ladle is in CounterTop_2", your `language_plan` must state: "Feedback indicates Ladle is at CounterTop_2, navigating there now."
   - **Action Alignment**: Your next action MUST be "find a CounterTop_2". Do not use generic names if a specific index is provided.
   - **Multiple Instances Handling**: If the environment contains multiple instances of a receptacle (e.g., several CounterTops), you must use the specific instance indicated by feedback. Failing to navigate to the correct instance (such as only using a generic "CounterTop") will result in the target object remaining invisible or inaccessible.
8. **Hand-State Awareness**: Before every "Pick up" action, verify your hand state in the history. If you are holding an object, the very next action in your `executable_plan` MUST be "Put down" or "Drop" to clear the gripper.
9. **Handle Non-Canonical Object Descriptions**: When the instruction refers to an object using a non-canonical or descriptive name (e.g., "wooden table"), and you are unsure which specific object it maps to in the environment, you should attempt to perform the required operation on all plausible candidate objects until the task succeeds or feedback clarifies the correct target.
10. **Never Output an Empty Plan**: If the task isn't explicitly confirmed as successful by feedback, continue planning. If you think it's done but no success message appears, assume a mistake was made and try a different approach.
11. **GPT-3.5 Strict Short-Plan Contract**:
   - Output at most 8 executable actions.
   - For counted placement tasks, output only one complete transfer cycle for the next object. After that object is placed, wait for the next planning call to handle the next object.
   - Do not output repeated probe lists such as find A, find B, find A, find B. Choose the single best next candidate from evidence.
   - If the current visual summary already shows the target object and its appearance matches the instruction, do not find that same object again; interact with it.
   - If history says you are holding an object, the next useful actions must place that held object at the goal destination before any new pickup, unless the instruction explicitly says to keep holding it while acting.
   - Before `open the X`, the immediately previous useful navigation must be `find a X`.
   - Before `put down the object in hand`, the immediately previous useful navigation must be the goal destination; for openable destinations such as Drawer, Cabinet, Fridge, Microwave, or Box, use `find destination -> open destination -> put down`.
   - For two/all/every counted object tasks, complete one full cycle at a time: find matching object -> pick it -> find destination -> open destination if needed -> put down; only then start the second object.
   - If you are already holding one counted candidate, do not find or pick another candidate. Output only the destination placement prefix now: `find destination -> open destination if needed -> put down`.
   - If feedback says an openable destination is already open, do not open it again; navigate to it if needed, then put down the held object.

The output json format should be {{"language_plan": str, "executable_plan": List[{{"action_id": int, "action_name": str, "predicted_state": str}}...]}}
For the "predicted_state" field, briefly describe what you expect to happen after executing this action (e.g., "Robot is now holding the apple", "Robot is at the CounterTop"). If unsure, write "Exploring environment."
!!! Please do not output anything other than the above-mentioned JSON, do not include ```json and ```!!!
'''


def _has_any(text: str, terms: list) -> bool:
    return any(term in text for term in terms)


def get_instruction_workflow_guidance(instruction: str) -> str:
    text = f" {instruction.lower().strip()} "
    rules = [
        "Before interacting with an object or receptacle, navigate with the matching `find` action for that target; if feedback provides an indexed name, use the exact indexed action only when it exists, otherwise use the unindexed target type from the action list.",
        "Before `put down the object in hand`, the immediately preceding useful navigation should be `find` for the intended destination; `drop` is not a substitute for goal placement.",
        "If feedback says the robot is already holding an object, do not plan another `pick up` until the held object has been intentionally placed or dropped.",
        "If feedback says an object is not visible because it is in/on another receptacle, first navigate to that receptacle; for goals that move the receptacle-with-contents, move the receptacle instead of trying to remove the contained object.",
        "Use `[Task Progress]` and feedback history to detect loops: if a successful-looking sequence does not increase progress, change object instance, destination instance, or workflow rather than repeating it."
    ]

    holding_action_goal = (
        "turn on" in text
        and (" with " in text or " while holding " in text or " as well as " in text)
    )
    lamp_inspection_goal = (
        _has_any(text, [" inspect ", " examine ", " look at ", " look over ", " check "])
        and _has_any(text, [" lamp ", " light ", " lit "])
    )
    if holding_action_goal or lamp_inspection_goal:
        rules.append(
            "For goals that require acting while holding another object, first pick up the required held object, keep it in hand, find the target device, then execute the target action; do not put the held object into/on the target."
        )
    if lamp_inspection_goal:
        rules.append(
            "For inspect/examine/look-at tasks involving lamp light, keep holding the inspected object, find the lamp, and turn it on if needed; a lamp is not a receptacle, so never put the held object on the lamp."
        )

    if not holding_action_goal and (" with " in text or " in it " in text or " on it " in text):
        rules.append(
            "For 'with' placement tasks, complete the containment relation first: pick the secondary object, find the main container object, put the secondary object into/on it, then pick the main object and place it at the destination."
        )
        rules.append(
            "In a phrase like 'main object with secondary object in/on it', never pick the main object first; if you are holding the main object while needing the secondary, put the main object down, pick the secondary, place it into/on the main object, and then stop touching the secondary."
        )
        rules.append(
            "If the history or feedback already shows the main object is holding the secondary object, skip rebuilding that relation; find the final destination and place the main object there."
        )
        rules.append(
            "If `[Task Progress]` increases after placing the secondary object into/on the main object, treat containment as complete: pick up the main object, find the final destination, and put it down there."
        )

    if _has_any(text, [" heated ", " heat ", " hot ", " warm ", " warmed ", " microwaved ", " cooked "]):
        rules.append(
            "For heated-object goals, create the heated state before final placement: find target, pick it up, find/open Microwave, place object inside, close Microwave, turn it on then off, reopen, retrieve the object, then continue."
        )

    if _has_any(text, [" cooled ", " cool ", " cold ", " chilled ", " chill "]):
        rules.append(
            "For cooled-object goals, create the cooled state before final placement: find target, pick it up, find/open Fridge, place object inside, close it briefly, reopen, retrieve the object, then continue."
        )

    if _has_any(text, [" sliced ", " slice ", " piece of ", " pieces of "]):
        rules.append(
            "For sliced-object goals, obtain a knife first, navigate to the target object, slice it, then pick up the resulting target object before final placement."
        )

    if _has_any(text, [" cleaned ", " clean ", " wash ", " washed ", " rinse ", " rinsed ", " rinsing "]) or " rinse off " in text:
        rules.append(
            "For cleaned/rinsed-object goals, clean before final placement: pick up the object, find a sink, put the object in the sink, turn the faucet on while the object is in the sink, turn the faucet off, then pick the object back up."
        )

    if "soap" in text and "bottle" not in text and "pump" not in text:
        rules.append(
            "For soap/bar-of-soap instructions that do not mention a bottle or pump, use SoapBar actions rather than SoapBottle actions."
        )

    if "under the sink" in text or "below the sink" in text or "cupboard" in text:
        rules.append(
            "For destinations under/below the sink or in a cupboard under the sink, use Cabinet actions as the executable receptacle."
        )

    if any(
        phrase in text
        for phrase in (
            "under the coffee maker",
            "below the coffee maker",
            "beneath the coffee maker",
            "under the coffee machine",
            "below the coffee machine",
        )
    ):
        rules.append(
            "For destinations under/below/beneath the coffee maker, use CounterTop as the executable placement surface; CoffeeMachine is the spatial reference, not the put-down receptacle."
        )

    if " bottle" in text or " bottles" in text:
        rules.append(
            "When an instruction says bottle/bottles without a precise class, try plausible ALFRED bottle classes one at a time; in bathroom or toilet/sink contexts prefer SprayBottle or SoapBottle before WineBottle or Glassbottle, and switch class after a cannot-find failure instead of retrying it."
        )

    if " glass " in text:
        rules.append(
            "When an instruction says glass as a held container or drinkware, consider Cup, Mug, or Glassbottle as plausible canonical classes; choose one valid class at a time and avoid unrelated pickup actions."
        )
        if " bowl" in text:
            rules.append(
                "When an instruction says glass bowl, use Bowl actions for the held container; glass is an appearance descriptor, not a request for Glassbottle."
            )

    if _has_any(text, [" jar ", " jars "]):
        rules.append(
            "For jar descriptions, try vessel-like classes such as Vase, Glassbottle, Cup, Mug, or Pot; do not choose Box or Cart unless feedback proves the vessel classes are absent."
        )

    if _has_any(text, [" soup ", " serving soup ", " serve soup "]):
        rules.append(
            "For functional descriptions about serving soup, prefer the utensil class Ladle before bowl-like containers unless the instruction explicitly asks for a bowl."
        )

    if _has_any(text, [" eating surface ", " hold your food ", " holds your food ", " dinner ", " meal "]):
        rules.append(
            "For functional descriptions such as eating surface, dinner surface, or something that holds food with utensils, prefer Plate as the main object before Bowl or other receptacles."
        )

    if _has_any(text, [" soft absorbent ", " absorbent fabric ", " towel ", " fabric "]):
        rules.append(
            "For soft absorbent fabric or towel-like descriptions, try HandTowel first in bathroom/tub contexts, then Cloth if HandTowel cannot be found."
        )

    if _has_any(text, [" rag ", " rags ", " cloth ", " cloths "]):
        rules.append(
            "For rag or cloth instructions, prefer Cloth actions first; use HandTowel only when the wording specifically says towel."
        )

    if _has_any(text, [" liquid container ", " closable liquid ", " closable container "]):
        rules.append(
            "For closable liquid-container descriptions, prefer SoapBottle, SprayBottle, Glassbottle, or WineBottle according to room context; try one valid bottle class at a time."
        )

    if _has_any(text, [" rice ", " cereal ", " soup bowl ", " eating vessel "]):
        rules.append(
            "For food-container or eating-vessel descriptions such as container for rice, cereal, or round eating vessel, prefer Bowl first, then Pot or Plate only if the wording suggests cookware or a flat eating surface."
        )

    if _has_any(text, [" reading material ", " newspaper ", " read ", " reading "]):
        rules.append(
            "For reading-material descriptions, prefer Newspaper before Book unless the instruction explicitly says book."
        )

    if _has_any(text, [" bound collection ", " bound narratives ", " narratives ", " book ", " books "]):
        rules.append(
            "For bound-collection or bound-narrative descriptions, prefer Book."
        )

    if _has_any(text, [" couch ", " sofa "]):
        rules.append(
            "For couch destinations, use Sofa actions."
        )

    if _has_any(text, [" place where you sleep ", " bed "]):
        rules.append(
            "For place-where-you-sleep descriptions, use Bed actions."
        )

    if _has_any(text, [" tv stand ", " television stand "]):
        rules.append(
            "For TV-stand destinations, use SideTable if no TVStand action exists."
        )

    if _has_any(text, [" communication device ", " phone ", " cell phone "]):
        rules.append(
            "For communication-device descriptions, prefer CellPhone."
        )

    if _has_any(text, [" changing channels ", " channel changer ", " remote control "]):
        rules.append(
            "For changing-channel device descriptions, prefer RemoteControl."
        )

    if _has_any(text, [" metal rack ", " rack ", " shelves "]):
        rules.append(
            "For rack or shelves destinations, prefer Shelf actions unless a more specific destination action is available."
        )

    if _has_any(text, [" jangling ", " keys ", " key ring "]):
        rules.append(
            "For jangling-item descriptions, prefer KeyChain."
        )

    if _has_any(text, [" music disc ", " music discs ", " disc ", " discs ", " music collection "]):
        rules.append(
            "For music-disc or music-collection descriptions, prefer CD."
        )

    if _has_any(text, [" flipping tool ", " spatula "]) or (" flip" in text and "tool" in text):
        rules.append(
            "For cooking flipping-tool descriptions, prefer Spatula."
        )

    if _has_any(text, [" beverage holder ", " drinking vessel ", " transparent beverage "]):
        rules.append(
            "For beverage-holder descriptions, prefer Cup or Glassbottle before unrelated containers."
        )

    if _has_any(text, [" cutting tool for butter ", " butter knife "]):
        rules.append(
            "For cutting-tool-for-butter descriptions, prefer ButterKnife."
        )

    if _has_any(text, [" green leaf ", " leafy ", " lettuce "]):
        rules.append(
            "For green leaf or leafy food descriptions, prefer Lettuce."
        )

    if _has_any(text, [" studying ", " study ", " desk "]):
        rules.append(
            "For studying-furniture descriptions, prefer Desk as the destination furniture."
        )

    if _has_any(text, [" black desk ", " white desk ", " desk "]):
        rules.append(
            "For colored desk destinations, use Desk actions; the color is a visual filter for the destination."
        )

    if _has_any(text, [" black table ", " white table ", " red table ", " green table ", " blue table ", " yellow table "]):
        rules.append(
            "For colored table destinations, use table actions such as DiningTable/CoffeeTable/SideTable according to the available action list; the color is a visual filter, not a separate object class."
        )

    if _has_any(text, [" white dresser ", " black dresser ", " dresser "]):
        rules.append(
            "For dresser destinations, use Dresser actions unless the instruction specifically says drawer."
        )

    if " drawer" in text or "drawers" in text:
        rules.append(
            "For drawer destinations, use Drawer actions. A dresser drawer is still a Drawer receptacle; Dresser is furniture context, not the openable container."
        )

    if (
        _has_any(text, [" all ", " every ", " each ", " two ", " three ", " pair "])
        or re.search(r"\b[2-9]\b", text)
        or _has_any(text, [" bottles ", " spray bottles ", " soap bottles ", " soap bars ", " bars of soap ", " tissue boxes ", " vases ", " jars "])
        or re.search(
            r"\b(?:red|green|blue|yellow|white|black|brown|gray|grey|silver|orange|purple|pink)\s+and\s+"
            r"(?:red|green|blue|yellow|white|black|brown|gray|grey|silver|orange|purple|pink)\s+"
            r"(?:vases?|jars?|bottles?|knives?|objects?|things?)\b",
            text,
        )
    ):
        rules.append(
            "For counted or all/every/each-object tasks, complete exactly one object instance at a time: find one candidate matching the description, pick it up, find the destination, place it, then repeat with a different instance until the requested count is complete; never pick a second instance while already holding the first."
        )
        rules.append(
            "In counted placement tasks, never plan two pickups before the first placement. The only valid pattern is one full transfer cycle, then the next full transfer cycle."
        )
        rules.append(
            "Prompting discipline for counted placement: output only the next transfer cycle, not the whole multi-object task. After the first object is placed, use the next planning call and updated history for the next object."
        )
        rules.append(
            "If history says you are already holding a counted candidate, the next plan must be only destination placement: find the requested destination, open it if it is openable, then put down; do not find or pick another candidate first."
        )
        rules.append(
            "After placing one counted instance at the destination, do not pick that placed object back up; search source receptacles or alternate instances for the next matching object."
        )
        rules.append(
            "When the action list contains indexed variants such as Object_2 or Object_3, use an untried indexed find action for the next counted instance instead of repeating the just-placed unindexed object."
        )
        rules.append(
            "If your reasoning says the next counted instance is on/in a source receptacle such as a desk, sink, counter, table, cabinet, or shelf, include that source navigation before the object search: find the source receptacle, then find/pick the object, then return to the destination."
        )
        rules.append(
            "If the only visible matching object is already at the goal destination after partial progress, treat it as already completed; do not pick it up. Move to another source area before searching for the next instance."
        )

    if _has_any(text, [" left ", " right ", " next to ", " beside ", " between ", " near ", " closest ", " farthest "]):
        rules.append(
            "For spatial-reference tasks, do not assume the first matching object is correct; use feedback, visible layout, and indexed instances to test plausible candidates systematically."
        )

    if _has_any(text, [" red ", " green ", " blue ", " yellow ", " white ", " black ", " brown ", " gray ", " grey ", " wooden ", " metal ", " small ", " large "]):
        rules.append(
            "For visual-appearance descriptions, infer plausible canonical object actions from the available action list, try one candidate class at a time, and do not output long probe lists of unrelated objects."
        )
        rules.append(
            "When a Current Visual Observation Summary is provided, treat its observed colors/materials as the current camera evidence: only interact with a visible candidate if it matches the requested appearance; otherwise inspect another indexed instance or plausible canonical class."
        )
        rules.append(
            "ALFRED visual noun mapping: bottle/bottles with color or pump usually map to SprayBottle or SoapBottle before WineBottle/Glassbottle; tissue boxes map to TissueBox; lamps map to DeskLamp or FloorLamp; small stand/statue-like objects map to Statue; vessels map to Vase."
        )
        rules.append(
            "For colored jar descriptions, try Vase or Glassbottle/Cup/Mug/Pot actions before Box or Cart; Box/Cart are containers/furniture, not jar-like vessels."
        )

    if "drawer" in text:
        rules.append(
            "For putting objects in a drawer, the destination micro-plan is mandatory: find a Drawer, open the Drawer, put down the object in hand. Do not use Dresser or Cabinet as the executable destination when Drawer actions exist."
        )
        rules.append(
            "If the Drawer has already been opened earlier in the same episode or feedback says it is already open, skip another open action and use: find a Drawer, put down the object in hand."
        )

    if _has_any(text, [" round table ", " brown shelf ", " bath tub ", " bathtub ", " tub ", " toilet tank ", " water tank "]):
        rules.append(
            "Treat visual or spatial adjectives on destinations as descriptors: round/white/black table still uses DiningTable, brown shelf uses Shelf, bath tub/tub uses Bathtub, and toilet or water tank uses Toilet."
        )

    if _has_any(text, [" in the box on ", " inside the box on ", " in a box on ", " inside a box on "]):
        rules.append(
            "For nested placement such as object in a box on a shelf, first put the object into the Box, then pick up the Box, navigate to the final shelf/table destination, and put down the Box."
        )

    return "## Task-Specific General Workflow Reminders\n" + "\n".join(
        f"- {rule}" for rule in rules
    )


WORLDMIND_ALFRED_SYSTEM_PROMPT2 ='''## You are an intelligent embodied agent operating in a home environment, equipped with an internal World Model.
You do not merely execute commands; you simulate the outcome of your actions before execution. For every step, you must think deeply about how your action will alter the environment to ensure the task is completed successfully. Your state prediction serves as a justification for your action—proving that you understand the consequences of your move.

## Action Descriptions and Validity Rules
• Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this skill is always valid
• Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.
• Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.
• Drop: Parameterized by the name of the object to put down. It is different from Put down action, as this does not guarantee the held object will be put into a specified receptacle.
• Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle.
• Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.
• Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object.
• Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object.
• Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.


## The available action id (0 ~ {}) and action names are: {}.

{}

## Guidelines
1. **Output Plan**: Avoid generating empty plan. Each plan should include no more than 20 actions.
2. **Visibility**: Always locate a visible object by the 'find' action before interacting with it.
3. **Action Guidelines**: Make sure match the action name and its corresponding action id in the output.\n Avoid performing actions that do not meet the defined validity criteria. For instance, if you want to put object in a receptacle, use 'put down' rather than 'drop' actions.
4. **Prevent Repeating Action Sequences**: Do not repeatedly execute the same action or sequence of actions.\n Try to modify the action sequence because previous actions do not lead to success.
5. **Multiple Instances**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3. You can explore these instances if you do not find the desired object in the current receptacle.
6. **Reflection on History and Feedback**: Use interaction history and feedback from the environment to refine and improve your current plan.\n If the last action is invalid, reflect on the reason, such as not adhering to action rules or missing preliminary actions, and adjust your plan accordingly.
7. **Dynamic Reasoning from Environment Feedback**: You must treat `env_feedback` as a direct instruction.
   - **Instruction Extraction**: If feedback says "Ladle is in CounterTop_2", your `language_plan` must state: "Feedback indicates Ladle is at CounterTop_2, navigating there now."
   - **Action Alignment**: Your next action MUST be "find a CounterTop_2". Do not use generic names if a specific index is provided.
   - **Multiple Instances Handling**: If the environment contains multiple instances of a receptacle (e.g., several CounterTops), you must use the specific instance indicated by feedback. Failing to navigate to the correct instance (such as only using a generic "CounterTop") will result in the target object remaining invisible or inaccessible.
8. **Handling "with" Placement Instructions**: When the instruction requires placing an object "with" another object (e.g., "set plate with a spoon in it on the kitchen table"), the workflow should be: first, find and pick up the "with" object (e.g., the spoon); then, find the main object (e.g., the plate) and put down the "with" object into/on it; next, pick up the main object (now containing the "with" object); finally, find the destination and put down the main object at the target location.
9. **General Placement Workflow**: For placement instructions, the workflow should be: first, find the target object; then, pick up the object; next, find the destination location; finally, use "put down the object in hand" to place the object at the destination.
10. **Infer and Try Candidate Objects**: When the instruction refers to an object, first infer which available action id objects in the current environment may correspond to the instruction. Then, attempt the required operation on each plausible candidate object until the task succeeds or feedback clarifies the correct target.
11. **Workflow for Object State Transformation Instructions**:
    If the instruction contains words like "warm", "heat", "microwaved", you MUST execute the heating workflow:
    find the target object → pick up the object → find a Microwave → open the Microwave → put down the object in hand → close the Microwave → turn on the Microwave → turn off the Microwave → open the Microwave → find the object → pick up the object → close the Microwave.
    If the instruction contains words like "cool", "cold", "chill", "fridge", you MUST execute the cooling workflow:
    find the target object → pick up the object → find a Fridge → open the Fridge → put down the object in hand → close the Fridge → open the Fridge → find the object → pick up the object → close the Fridge.
    If the instruction contains words like "slice", "piece of", you MUST execute the slicing workflow:
    find a Knife → pick up the Knife → find the object to slice → slice the object → find a CounterTop → put down the object in hand → find the sliced object → pick up the sliced object.
    If the instruction contains words like "clean", "cleaned", "wash", you MUST execute the cleaning workflow:
    pick up the object → find a Sink → put down the object in hand into the sink → find the Faucet → turn on the Faucet → turn off the Faucet → pick up the object again.
    **Note:** No object is ever already heated, cooled, sliced, or cleaned in the environment. If the instruction refers to an "already" processed object (e.g., "microwaved bread", "sliced apple", "cold lettuce", "cleaned plate"), you MUST perform the corresponding workflow to achieve that state before proceeding to the next step.
12. **Prediction-Driven Plan Correction**: Do not let state prediction distract you; use it to VALIDATE your plan. Mentally simulate the state change for each action. If the predicted state implies a gap in logic (e.g., the object is predicted to be 'dirty' because you missed the 'turn on faucet' step), you MUST immediately correct your action sequence to include the missing steps. Ensure the physical chain of causality (Action -> State Change) is complete before outputting the plan.
13. **Destination and Object Awareness Before Actions**: Before executing the "put down the object in hand" action, you must use "find" to explicitly navigate to the intended destination receptacle. Similarly, before executing a "pick up" action, you must use "find" to explicitly navigate to the target object. This ensures you are physically close to the correct object or location, avoiding placement and pickup errors.
14. **Strict Step-by-Step Physical Reality**: Imagine you are physically acting in a real world. You MUST execute the plan strictly step-by-step. **DO NOT SKIP STEPS**. For example, you cannot 'Pick up' an object without navigating to it first, and you cannot 'Put down' an object without navigating to the destination first. Teleportation or action skipping leads to immediate physical failure.

The output json format should be {{"language_plan": str, "executable_plan": List[{{"action_id": int, "action_name": str, "predicted_state": str}}...]}}
For the "predicted_state" field:
If the target object or destination is NOT VISIBLE (Exploration Phase), you MUST output exactly the string: "Exploration phase: target not visible, prediction skipped."
Otherwise, describe the specific environmental change.
!!! Please do not output anything other than the above-mentioned JSON, do not include ```json and ```!!!
'''


WorldMind_TEMPLATE = '''
The output json format should be {{"language_plan": str, "executable_plan": List[{{"action_id": int, "action_name": str, "predicted_state": str}}...]}}
The fields in the above JSON follow the purpose below:
1. language_plan is for your Chain-of-Thought.
2. executable_plan is a list of concrete actions to be executed.
!!! Please do not output anything other than the above-mentioned JSON, do not include ```json and ```!!!
'''

WorldMind_TEMPLATE_LANG = '''
The output json format should be {{"language_plan": str, "executable_plan": List[{{"action_id": int, "action_name": str, "predicted_state": str}}...]}}
The fields in the above JSON follow the purpose below:
1. language_plan is for your Chain-of-Thought.
2. executable_plan is a list of concrete actions to be executed.
!!! Please do not output anything other than the above-mentioned JSON, do not include ```json and ```!!!
'''


def get_worldmind_system_prompt(num_actions: int, available_action_str: str, examples_str: str = "", include_workflow: bool = True) -> str:
    base_prompt = WORLDMIND_ALFRED_SYSTEM_PROMPT.format(num_actions, available_action_str, examples_str)
    return base_prompt


def get_worldmind_examples(n_shot: int = 5) -> list:
    return []


def format_worldmind_examples(examples: list) -> str:
    return '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i, x in enumerate(examples)])



import re
import json

def fix_json_worldmind(json_str):
    if not json_str or not json_str.strip():
        return ""
    json_str = json_str.strip()
    if json_str.startswith("```"):
        json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        json_str = json_str.strip()
    json_str = re.sub(r'\}\s*\n\s*"', '},\n"', json_str)
    json_str = re.sub(r'\}\s*\n\s*\{', '},\n{', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    json_str = re.sub(r',\s*\}', '}', json_str)
    return json_str


def parse_json_worldmind(json_str):
    if not json_str or not json_str.strip():
        raise ValueError("Empty input string")
    json_str = fix_json_worldmind(json_str)
    first_brace = json_str.find('{')
    if first_brace == -1:
        raise ValueError("No opening brace found")
    brace_count = 0
    last_brace = -1
    in_string = False
    escape_next = False
    for i in range(first_brace, len(json_str)):
        char = json_str[i]
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_brace = i
                    break
    if last_brace == -1:
        raise ValueError("No matching closing brace found")
    json_text = json_str[first_brace:last_brace+1]
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")
