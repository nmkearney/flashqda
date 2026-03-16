Context:

Consider the following {granularity}(s), which occur before the {granularity} to be analyzed:  
{context_window}

---

Task:

Read the following {granularity}:  
{item}

Now focus your analysis on the following cause-effect pair extracted from that {granularity}:
{pair}

Determine whether any of the following labels apply to this specific pair:
{label_list}

If none apply, return "none".

---

Instructions:

1. Use only the label names, not the description.
2. Use only the label names listed above. Do not invent new ones.
3. Return your answer in a single JSON object, like this:

{{
  "labels": ["label 1", "label 2"],
  "reasoning": {{
    "label 1": "Brief 1-2 sentences why label 1 applies.",
    "label 2": "Brief 1-2 sentences why label 2 applies."
  }}
}}

4. Include the "reasoning" object ONLY if at least one label applies.