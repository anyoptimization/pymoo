# Command Execution Instructions

## Overview

When the user asks you to execute a command, look for the corresponding file in the `.ai/commands/` directory and follow the instructions contained within that file.

## Command Execution Process

1. **Command Request**: Commands can be triggered in multiple ways:
   - Natural language: "run fix-tests", "execute fix-examples"
   - Slash commands: `/fix-tests`, `/fix-examples`, `/fix-docs`
   - Direct command names: "fix-tests", "fix-examples"

2. **Locate Command File**: Find the corresponding `.md` file in `.ai/commands/` directory:
   - Command: "fix-tests" → File: `.ai/commands/fix-tests.md`
   - Command: "fix-examples" → File: `.ai/commands/fix-examples.md`
   - Command: "fix-docs" → File: `.ai/commands/fix-docs.md`

3. **Read and Execute**: Read the command file and follow the step-by-step instructions provided.

4. **Handle Arguments**: Commands may contain `$ARGUMENTS` placeholders:
   - **Arguments Provided**: If arguments are already provided in the user's prompt, substitute them directly
   - **Arguments Missing**: If arguments are needed but not provided, ask the user for clarification in a follow-up question

## Example Usage

**User Request**: "Execute fix-tests for the algorithms module"
- **Action**: Read `.ai/commands/fix-tests.md`
- **Arguments**: `$ARGUMENTS = "algorithms module"` (provided in prompt)
- **Execution**: Follow the instructions in the file, substituting the arguments where needed

**User Request**: "Run fix-examples"
- **Action**: Read `.ai/commands/fix-examples.md`
- **Arguments**: If the command file requires `$ARGUMENTS` but none provided, ask user: "Which examples would you like me to focus on?"

## Command File Structure

Command files in `.ai/commands/` typically contain:
- Step-by-step instructions
- Code snippets to execute
- Conditional logic based on results
- Argument placeholders (`$ARGUMENTS`)
- Expected outcomes and next steps

## Best Practices

- Always read the entire command file before starting execution
- Follow the instructions sequentially
- Substitute arguments appropriately
- If a step fails, continue with any error handling instructions provided in the command file
- Report progress and results back to the user as you execute each major step