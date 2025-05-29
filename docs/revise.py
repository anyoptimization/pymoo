import os
import re
from typing import Optional

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def apply_diff(original_text: str, diff_output: str) -> str:
    # Pattern to match diff blocks
    diff_pattern = r'<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE'

    # Find all diff blocks
    diff_blocks = re.findall(diff_pattern, diff_output, re.DOTALL)

    modified_text = original_text

    for search_text, replace_text in diff_blocks:
        # Strip leading/trailing whitespace but preserve internal formatting
        search_text = search_text.strip()
        replace_text = replace_text.strip()

        # Check if the search text exists in the original
        if search_text in modified_text:
            # Replace the first occurrence
            modified_text = modified_text.replace(search_text, replace_text, 1)
            print(f"✓ Applied\n'{search_text}'\n'{replace_text}'")
        else:
            print(f"✗ Warning: Could not find: '{search_text}'")

    return modified_text

class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)


OPENROUTER_API_KEY = '__PLACEHOLDER__'

llm = ChatOpenRouter(
    model_name="anthropic/claude-3.7-sonnet",
    openai_api_key=OPENROUTER_API_KEY
)

SYSTEM = """
# System Prompt

You are a proofreader. Fix only typos and grammar errors in the given text. Do not change technical terms, code, or content meaning. Output corrections as diff blocks using this format:

```
<<<<<<< SEARCH 
The algorithim processes data and return results.
=======
The algorithm processes data and returns results.
>>>>>>> REPLACE 
```

If no corrections are needed, respond with an empty string.
"""

HUMAN = """Please review the following MyST markdown text for typos and grammar errors. Provide corrections in diff format:


---

# Example Usage

**Input text:**
```markdown
# Data Processing

The algorithim processes data and return results to the user. It's performance is optimal for large datasets.
```

**AI Response:**
```
<<<<<<< SEARCH 
The algorithim processes data and return results to the user.
=======
The algorithm processes data and returns results to the user.
>>>>>>> REPLACE 

<<<<<<< SEARCH 
It's performance is optimal for large datasets.
=======
Its performance is optimal for large datasets.
>>>>>>> REPLACE 
```

I will apply later the diff you are outputting. Let me give you now the text:

```markdown
{text}
```

"""


def proof_read(path):
    template = ChatPromptTemplate([
        ("system", SYSTEM),
        ("human", HUMAN),
    ])

    with open(path) as f:
        text = f.read()

    messages = template.format_messages(text=text)

    response = llm.invoke(messages)

    diff_output = response.content

    mod_text = apply_diff(text, diff_output)

    with open(path, "w") as f:
        f.write(mod_text)


from tests.test_util import files_from_folder, DOCS

paths = files_from_folder(DOCS, regex='**/*.md')

for path in paths:
    print("_______________")
    print(path)
    proof_read(path)
