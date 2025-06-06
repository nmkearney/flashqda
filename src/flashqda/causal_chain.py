import os
import csv
from collections import defaultdict
from .embedding_utils import update_embeddings_from_data, compute_similarity_matrix, cosine_similarity

class CausalChain:
    def __init__(self, data_list=None, embeddings_path=None, data_path=None):
        if data_list is not None:
            self.data_list = data_list
        elif data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.data_list = list(reader)
        else:
            raise ValueError("Either data_list or data_path must be provided")
        self.embeddings_path = embeddings_path

        self.embeddings, self.changed_concepts = update_embeddings_from_data(self.data_list, embeddings_path)
        self.similarity_matrix = compute_similarity_matrix(self.embeddings)

        self.step_id_counter = self._init_step_id_counter()
        self.existing_pairs = set((entry.get("cause"), entry.get("effect")) for entry in self.data_list)

    def _init_step_id_counter(self):
        max_id = 0
        for entry in self.data_list:
            if "id" in entry:
                try:
                    max_id = max(max_id, int(entry["id"]))
                except ValueError:
                    continue
        return max_id + 1

    def suggest_causal_stitches(self, threshold=0.75, top_n=3, auto_export_path=None):
        concepts = set()
        for entry in self.data_list:
            if entry.get("cause"):
                concepts.add(entry["cause"].strip())
            if entry.get("effect"):
                concepts.add(entry["effect"].strip())

        concept_list = list(concepts)
        similarities = []

        for i, c1 in enumerate(concept_list):
            for j, c2 in enumerate(concept_list):
                if i >= j:
                    continue
                e1 = self.embeddings.get(f"cause::{c1}") or self.embeddings.get(f"effect::{c1}")
                e2 = self.embeddings.get(f"cause::{c2}") or self.embeddings.get(f"effect::{c2}")
                if not e1 or not e2:
                    continue
                sim = cosine_similarity(e1["embedding"], e2["embedding"])
                if sim >= threshold:
                    similarities.append((c1, c2, sim))

        # dedupe: for each concept, keep top_n matches
        match_map = defaultdict(list)
        for c1, c2, sim in similarities:
            match_map[c1].append((c2, sim))
            match_map[c2].append((c1, sim))

        deduped_matches = set()
        for c, matches in match_map.items():
            top_matches = sorted(matches, key=lambda x: -x[1])[:top_n]
            for m, score in top_matches:
                pair = tuple(sorted([c, m]))
                deduped_matches.add((pair[0], pair[1], score))

        # now identify rows using these stitched concepts
        suggestions = []
        suggestion_id = 1
        for cause1, cause2, score in deduped_matches:
            rows1 = [r for r in self.data_list if r.get("effect") == cause1]
            rows2 = [r for r in self.data_list if r.get("cause") == cause2]
            for r1 in rows1:
                for r2 in rows2:
                    if r1 is r2:
                        continue
                    suggestions.append({
                        "suggestion_id": suggestion_id,
                        "from_id": f"{r1.get('document_id', '')}_{r1.get('sentence_id', '')}_{r1.get('relationship_id', r1.get('id', ''))}",
                        "from_sentence": r1.get("sentence", ""),
                        "from_cause": r1.get("cause", ""),
                        "from_effect": r1.get("effect", ""),
                        "to_id": f"{r2.get('document_id', '')}_{r2.get('sentence_id', '')}_{r2.get('relationship_id', r2.get('id', ''))}",
                        "to_sentence": r2.get("sentence", ""),
                        "to_cause": r2.get("cause", ""),
                        "to_effect": r2.get("effect", ""),
                        "similarity": score,
                        "accepted": "",
                        "analyst_comment": "",
                        "status": "valid"
                    })
                    suggestion_id += 1

            if auto_export_path:
                self.export_suggestions_csv(suggestions, auto_export_path)

        return suggestions

    def validate(self, suggestions, add_new=True):
        updated_suggestions = []
        current_ids = {str(row.get("id")) for row in self.data_list if row.get("id")}
        row_lookup = {str(row.get("id")): row for row in self.data_list if row.get("id")}
        for s in suggestions:
            from_id = str(s.get("from_id"))
            to_id = str(s.get("to_id"))
            from_row = row_lookup.get(from_id)
            to_row = row_lookup.get(to_id)
            if not from_row or not to_row:
                s["status"] = "broken"
            elif (from_row.get("cause") != s.get("from_cause") or from_row.get("effect") != s.get("from_effect") or
                  to_row.get("cause") != s.get("to_cause") or to_row.get("effect") != s.get("to_effect")):
                s["status"] = "stale"
            else:
                s["status"] = "valid"
            updated_suggestions.append(s)

        if add_new:
            fresh = self.suggest_causal_stitches()
            existing_keys = {(s["from_id"], s["to_id"]) for s in updated_suggestions}
            for s in fresh:
                key = (s["from_id"], s["to_id"])
                if key not in existing_keys:
                    updated_suggestions.append(s)

        return updated_suggestions

    def validate_from_file(self, input_path, output_path, add_new=True):
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            suggestions = list(reader)

        updated_suggestions = self.validate(suggestions, add_new=add_new)
        self.export_suggestions_csv(updated_suggestions, output_path)

    
    def export_suggestions_csv(self, suggestions, output_path):
        fieldnames = [
            "suggestion_id", "from_sentence", "from_cause", "from_effect",
            "to_cause", "to_effect", "to_sentence",
            "from_id", "to_id", "similarity", "accepted", "analyst_comment", "status"
        ]
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for suggestion in suggestions:
                writer.writerow(suggestion)

    def export_data(self, output_path):
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data_list, f, indent=2, ensure_ascii=False)
