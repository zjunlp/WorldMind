"""
WorldMind Plugin Example

This example demonstrates how to use the three independent modules of WorldMind:

1. ProcessExperienceModule: For real-time experience extraction from prediction errors
2. GoalExperienceModule: For extracting experience from successful task trajectories
3. ExperienceRetrievalModule: For retrieving and injecting experiences into agent prompts

Each module has clear input/output specifications and can be used independently.
"""

from worldmind_plugin import (
    WorldMindConfig,
    ProcessExperienceModule,
    GoalExperienceModule,
    ExperienceRetrievalModule,
    ProcessTrajectoryStep,
    GoalTrajectoryStep,
    PREDICTION_CONSTRAINT_PROMPT,
    EXPLORATION_PHASE_MARKER
)


# =============================================================================
# Configuration
# =============================================================================

def create_config():
    """Create WorldMind configuration."""
    config = WorldMindConfig(
        # API Configuration
        api_key="your-api-key-here",
        api_base="https://api.openai.com/v1",  # Or your custom endpoint
        
        # Multimodal Configuration
        is_multimodal=False,  # Set True if using vision capabilities
        
        # Model Configuration
        discriminator_model="gpt-4o-mini",
        reflector_model="gpt-4o-mini",
        summarizer_model="gpt-4o",  # Vision model for multimodal
        extractor_model="gpt-4o-mini",
        refiner_model="gpt-4o-mini",
        
        # Experience Configuration
        enable_experience_refine=True,
        goal_experience_top_k=3,
        process_experience_top_k=5,
        goal_trajectory_include_observation=True,  # Include observation in goal trajectory
        
        # Feedback Configuration
        use_env_feedback=True,
        
        # Save Configuration
        save_path="./worldmind_output",
        
        # Output Configuration
        detailed_output=True
    )
    return config


# =============================================================================
# Module 1: Process Experience Module
# =============================================================================

def example_process_experience_module():
    """
    Example: Using ProcessExperienceModule for real-time experience extraction.
    
    This module is used during task execution to:
    1. Compare agent's predicted state with actual observation
    2. Reflect on prediction errors
    3. Extract and save process experiences
    
    Input:
        - task_instruction: str - The task being performed
        - trajectory: List[ProcessTrajectoryStep] - Each step contains:
            - observation: str - The current state/observation
            - action: str - The action executed
            - predicted_state: str - What the agent predicted would happen
            - env_feedback: str - Feedback from the environment
    
    Output:
        - List[str] - Extracted experience entries
    
    Storage Format:
        {
            "instruction": "task instruction",
            "knowledge": ["experience 1", "experience 2", ...]
        }
    """
    print("\n" + "="*60)
    print("Module 1: ProcessExperienceModule")
    print("="*60)
    
    config = create_config()
    process_module = ProcessExperienceModule(config)
    
    # Define task instruction
    task_instruction = "Complete the user registration form and submit it"
    
    # Build trajectory - each step has observation, action, predicted_state, env_feedback
    trajectory = [
        ProcessTrajectoryStep(
            observation="Registration form page loaded. All fields are empty.",
            action="Fill username field with 'testuser'",
            predicted_state="Username field contains 'testuser'",
            env_feedback="Field updated successfully"
        ),
        ProcessTrajectoryStep(
            observation="Username field shows 'testuser'. Other fields empty.",
            action="Fill email field with 'test@example'",
            predicted_state="Email field contains valid email",
            env_feedback="Warning: Invalid email format"
        ),
        ProcessTrajectoryStep(
            observation="Email field shows 'test@example' with error indicator.",
            action="Submit form",
            predicted_state="Form submitted successfully, redirected to success page",
            env_feedback="Error: Please fix validation errors before submitting"
        )
    ]
    
    # Process the entire trajectory at once
    experiences = process_module.process_trajectory(
        task_instruction=task_instruction,
        trajectory=trajectory
    )
    
    print(f"\nTask: {task_instruction}")
    print(f"Trajectory steps: {len(trajectory)}")
    print(f"Extracted experiences: {len(experiences)}")
    for exp in experiences:
        print(f"  - {exp}")
    
    # Alternative: Process steps one by one (for real-time use)
    print("\n--- Real-time processing example ---")
    action_history = []
    
    for i, step in enumerate(trajectory):
        has_error, step_experiences = process_module.process_single_step(
            task_instruction=task_instruction,
            step=step,
            action_history=action_history,
            state_before=trajectory[i-1].observation if i > 0 else step.observation
        )
        
        if has_error:
            print(f"Step {i+1}: Prediction error detected!")
            for exp in step_experiences:
                print(f"  Experience: {exp}")
        else:
            print(f"Step {i+1}: Prediction matched or skipped")
        
        action_history.append(f"Action: {step.action}, Feedback: {step.env_feedback}")


# =============================================================================
# Module 2: Goal Experience Module
# =============================================================================

def example_goal_experience_module():
    """
    Example: Using GoalExperienceModule for extracting success patterns.
    
    This module is used when a task is successfully completed to:
    1. Analyze the successful trajectory
    2. Extract reusable workflow/experience
    3. Save goal experience paired with instruction
    
    Input:
        - task_instruction: str - The completed task
        - trajectory: List[GoalTrajectoryStep] - Each step contains:
            - action: str - The action executed
            - env_feedback: str - Feedback from the environment
            - observation: str (optional) - State observation
    
    Output:
        - str - Extracted goal experience
    
    Storage Format:
        {
            "instruction": "task instruction",
            "goal_experience": "extracted workflow description"
        }
    """
    print("\n" + "="*60)
    print("Module 2: GoalExperienceModule")
    print("="*60)
    
    config = create_config()
    goal_module = GoalExperienceModule(config)
    
    # Define successful task
    task_instruction = "Send an email with attachment to recipient@example.com"
    
    # Build trajectory - action and env_feedback, optionally observation
    trajectory = [
        GoalTrajectoryStep(
            action="Open email client",
            env_feedback="Email client opened successfully",
            observation="Email client main window displayed"  # Optional
        ),
        GoalTrajectoryStep(
            action="Click 'Compose' button",
            env_feedback="New email draft opened",
            observation="Empty email composition form shown"
        ),
        GoalTrajectoryStep(
            action="Enter recipient 'recipient@example.com'",
            env_feedback="Recipient added",
            observation=""  # Observation can be empty
        ),
        GoalTrajectoryStep(
            action="Enter subject 'Monthly Report'",
            env_feedback="Subject set"
        ),
        GoalTrajectoryStep(
            action="Click 'Attach file' button",
            env_feedback="File browser opened"
        ),
        GoalTrajectoryStep(
            action="Select 'report.pdf' and confirm",
            env_feedback="File attached: report.pdf (2.5 MB)"
        ),
        GoalTrajectoryStep(
            action="Click 'Send' button",
            env_feedback="Email sent successfully"
        )
    ]
    
    # Extract experience (call this when task succeeds)
    experience = goal_module.extract_experience(
        task_instruction=task_instruction,
        trajectory=trajectory
    )
    
    print(f"\nTask: {task_instruction}")
    print(f"Trajectory steps: {len(trajectory)}")
    print(f"Include observation: {goal_module.include_observation}")
    print(f"\nExtracted experience:")
    print(f"  {experience}")
    
    # You can control whether to include observation
    goal_module.set_include_observation(False)
    print(f"\nWith include_observation=False, trajectory would not include observations")


# =============================================================================
# Module 3: Experience Retrieval Module
# =============================================================================

def example_experience_retrieval_module():
    """
    Example: Using ExperienceRetrievalModule for experience injection.
    
    This module is used before task execution to:
    1. Retrieve relevant goal experiences (success patterns)
    2. Retrieve relevant process experiences (error lessons)
    3. Optionally refine/consolidate experiences
    4. Format experiences for injection into agent prompt
    
    Input:
        - task_instruction: str - The current task
    
    Output:
        - Dict containing:
            - goal_experiences: List[Dict] - Retrieved goal experiences
            - process_experiences: List[Dict] - Retrieved process experiences
            - refined_experience: Dict (optional) - Consolidated experience
            - formatted_prompt: str - Ready-to-inject prompt text
    """
    print("\n" + "="*60)
    print("Module 3: ExperienceRetrievalModule")
    print("="*60)
    
    config = create_config()
    retrieval_module = ExperienceRetrievalModule(config)
    
    # Current task instruction
    task_instruction = "Send a document via email to john@example.com"
    
    # Retrieve experiences
    result = retrieval_module.retrieve(
        task_instruction=task_instruction,
        enable_refine=True  # Override config setting if needed
    )
    
    print(f"\nTask: {task_instruction}")
    print(f"\nRetrieved goal experiences: {len(result['goal_experiences'])}")
    for exp in result['goal_experiences']:
        print(f"  - From: {exp.get('instruction', 'Unknown')[:50]}...")
        print(f"    Experience: {exp.get('goal_experience', '')[:100]}...")
    
    print(f"\nRetrieved process experiences: {len(result['process_experiences'])}")
    for exp in result['process_experiences']:
        print(f"  - {exp.get('knowledge', '')[:100]}...")
    
    if result['refined_experience']:
        print(f"\nRefined experience:")
        print(f"  Merged: {result['refined_experience'].get('merged_experience', '')[:200]}...")
        print(f"  Plan: {result['refined_experience'].get('initial_plan', '')[:200]}...")
    
    print(f"\nFormatted prompt length: {len(result['formatted_prompt'])} chars")
    
    # Get statistics
    print(f"\nExperience counts:")
    print(f"  Goal experiences: {retrieval_module.get_goal_experience_count()}")
    print(f"  Process experiences: {retrieval_module.get_process_experience_count()}")
    
    # Reload experiences (after new experiences are saved by other modules)
    retrieval_module.reload_experiences()


# =============================================================================
# Complete Integration Example
# =============================================================================

def example_full_integration():
    """
    Example: Complete integration in an agent loop.
    
    This shows how to integrate all three modules in a typical agent workflow:
    1. Before task: Retrieve experiences
    2. During task: Extract process experiences from errors
    3. After success: Extract goal experience
    """
    print("\n" + "="*60)
    print("Complete Integration Example")
    print("="*60)
    
    config = create_config()
    
    # Initialize modules
    process_module = ProcessExperienceModule(config)
    goal_module = GoalExperienceModule(config)
    retrieval_module = ExperienceRetrievalModule(config)
    
    # =========================================================================
    # Phase 1: Before Task - Retrieve Experiences
    # =========================================================================
    
    task_instruction = "Book a flight from NYC to LA for next Monday"
    
    print(f"\n[Phase 1] Retrieving experiences for: {task_instruction}")
    
    # Get experiences to inject into agent prompt
    experiences = retrieval_module.retrieve(task_instruction)
    
    # Build agent system prompt
    agent_system_prompt = """You are a helpful assistant that helps users book flights.

{prediction_constraint}

{experiences}
""".format(
        prediction_constraint=PREDICTION_CONSTRAINT_PROMPT,
        experiences=experiences['formatted_prompt']
    )
    
    print(f"Agent prompt includes {len(experiences['goal_experiences'])} goal experiences")
    print(f"Agent prompt includes {len(experiences['process_experiences'])} process experiences")
    
    # =========================================================================
    # Phase 2: During Task - Process Steps
    # =========================================================================
    
    print(f"\n[Phase 2] Processing task steps...")
    
    # Simulated agent execution
    steps = [
        ProcessTrajectoryStep(
            observation="Flight search page loaded",
            action="Search flights NYC to LA",
            predicted_state=EXPLORATION_PHASE_MARKER,  # Exploration phase
            env_feedback="Searching..."
        ),
        ProcessTrajectoryStep(
            observation="Search results showing 5 flights",
            action="Select cheapest flight",
            predicted_state="Flight selected, booking form displayed",
            env_feedback="Flight AA123 selected"
        ),
        ProcessTrajectoryStep(
            observation="Booking form displayed",
            action="Submit booking",
            predicted_state="Booking confirmed successfully",
            env_feedback="Error: Please select a date first"  # Error!
        )
    ]
    
    # Track trajectory for goal experience
    goal_trajectory = []
    action_history = []
    
    for i, step in enumerate(steps):
        print(f"\n  Step {i+1}: {step.action}")
        
        # Process for errors
        has_error, step_experiences = process_module.process_single_step(
            task_instruction=task_instruction,
            step=step,
            action_history=action_history
        )
        
        if has_error:
            print(f"    -> Prediction error! Extracted {len(step_experiences)} experiences")
        elif step.predicted_state == EXPLORATION_PHASE_MARKER:
            print(f"    -> Exploration phase, skipped")
        else:
            print(f"    -> Prediction matched")
        
        # Track for goal experience
        goal_trajectory.append(GoalTrajectoryStep(
            action=step.action,
            env_feedback=step.env_feedback,
            observation=step.observation
        ))
        
        action_history.append(f"Action: {step.action}, Feedback: {step.env_feedback}")
    
    # =========================================================================
    # Phase 3: After Success - Extract Goal Experience
    # =========================================================================
    
    task_success = True  # Assume task eventually succeeded
    
    if task_success:
        print(f"\n[Phase 3] Task succeeded! Extracting goal experience...")
        
        experience = goal_module.extract_experience(
            task_instruction=task_instruction,
            trajectory=goal_trajectory
        )
        
        print(f"  Extracted: {experience[:100] if experience else 'None'}...")
        
        # Reload retrieval module to include new experiences
        retrieval_module.reload_experiences()
    
    print("\n[Complete] All phases finished successfully!")


# =============================================================================
# Trajectory Format Reference
# =============================================================================

def print_trajectory_format_reference():
    """Print reference for trajectory formats."""
    print("\n" + "="*60)
    print("Trajectory Format Reference")
    print("="*60)
    
    print("""
## ProcessTrajectoryStep (for ProcessExperienceModule)

Each step contains:
- observation: str     # Current state/observation text
- action: str          # Action executed (name or description)
- predicted_state: str # Agent's prediction of state after action
- env_feedback: str    # Environment's feedback after action

Example:
    ProcessTrajectoryStep(
        observation="Login page displayed. Username and password fields visible.",
        action="Enter username 'admin'",
        predicted_state="Username field shows 'admin'",
        env_feedback="Field updated"
    )

Note: If predicted_state equals EXPLORATION_PHASE_MARKER, discrimination is skipped.


## GoalTrajectoryStep (for GoalExperienceModule)

Each step contains:
- action: str          # Action executed
- env_feedback: str    # Environment's feedback (can be empty)
- observation: str     # Optional observation (included if config allows)

Example:
    GoalTrajectoryStep(
        action="Click submit button",
        env_feedback="Form submitted successfully",
        observation="Success message displayed"  # Optional
    )


## Storage Formats

Process Experience (process_experiences.json):
{
    "instruction": "Complete user registration",
    "knowledge": [
        "Experience: Form submission failed because required field was empty. Lesson: Always validate required fields.",
        "Experience: Email validation failed because format was invalid. Lesson: Use proper email format."
    ]
}

Goal Experience (goal_experiences.json):
{
    "instruction": "Send email with attachment",
    "goal_experience": "To send an email with attachment: (1) Open compose form, (2) Fill recipient and subject, (3) Use attach button to add file, (4) Verify attachment appears, (5) Click send. Success: Email sent confirmation displayed."
}
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("WorldMind Plugin Examples")
    print("="*60)
    print("This demonstrates the three independent modules:\n")
    print("1. ProcessExperienceModule - Extract experiences from prediction errors")
    print("2. GoalExperienceModule - Extract experiences from successful tasks")
    print("3. ExperienceRetrievalModule - Retrieve and inject experiences\n")
    
    # Print trajectory format reference
    print_trajectory_format_reference()
    
    # Note: Uncomment the examples below to run them
    # (Requires valid API key in configuration)
    
    # example_process_experience_module()
    # example_goal_experience_module()
    # example_experience_retrieval_module()
    # example_full_integration()
    
    print("\nTo run examples, set your API key and uncomment the example calls.")
