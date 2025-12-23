import sys
import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# --- 1. HELPER FUNCTIONS ---

def is_vendor_file(file_path: str) -> bool:
    if not file_path: return False
    vendor_patterns = [
        r"^node_modules/", r"^@vue/", r"vuetify/", r"core-js", r"zone\.js",
        r"runtime-core\.esm-bundler\.js", r"reactivity\.esm-bundler\.js",
        r"proxiedModel\.mjs", r"app\..*\.js", r"vendor\..*\.js", r"chunk-.*\.js",
        r"LogbackBugsnagAppender\.java", r"AppenderBase\.java",
        r"AppenderAttachableImpl\.java", r"Logger\.java", r"ch\.qos\.logback\.",
        r"org\.springframework\.", r"org\.apache\.commons\.", r"java\.util\.",
        r"javax?\.", r"sun\.reflect\."
    ]
    return any(re.search(p, file_path) for p in vendor_patterns)

def parse_stack_frames(input_str: Any) -> List[Dict[str, Any]]:
    if not input_str or not isinstance(input_str, str): return []
    frames = []
    seen = set()

    # HTML format: <strong>file:line</strong> - code
    html_matches = re.findall(r'<strong>(.*?)</strong>', input_str)
    for raw in html_matches:
        raw = raw.strip()
        if raw in seen: continue
        seen.add(raw)
        
        # Handle "file:line - code"
        parts = raw.split(' - ', 1)
        file_line = parts[0]
        
        last_colon = file_line.rfind(':')
        if last_colon == -1: continue
        
        file_path = file_line[:last_colon]
        line_str = file_line[last_colon+1:]
        file_name = file_path.split('/')[-1]
        
        frames.append({
            'file': file_name,
            'line': int(line_str) if line_str.isdigit() else None,
            'vendor': is_vendor_file(file_path),
            'full_path': file_path
        })

    # Fallback: Plain text format
    if not frames:
        plain_matches = re.findall(r'([\w@:\/\.\-]+?\.(?:vue|js|ts|mjs|jsx|tsx|java)):(\d+)', input_str)
        for full_path, line_num in plain_matches:
            raw = f"{full_path}:{line_num}"
            if raw in seen: continue
            seen.add(raw)
            
            file_name = full_path.split('/')[-1]
            frames.append({
                'file': file_name,
                'line': int(line_num),
                'vendor': is_vendor_file(full_path),
                'full_path': full_path
            })
            
    return frames

# --- 2. NORMALIZATION ---

def normalize_veoci_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts fields from Veoci API format (nested in 'values') into Canonical Schema"""
    values = entry.get('values', {})
    
    def get_val(key):
        field = values.get(key, {})
        if not field: return None
        val = field.get('data', {}).get('value')
        if isinstance(val, list): return ", ".join(val) # Handle multi-value
        return val

    return {
        'entry_id': str(entry.get('id')),
        'project': get_val('21'),          # Project ID
        'release_stage': get_val('25'),    # Release Stage
        'app_version': get_val('18'),      # App Version
        'timestamp': entry.get('lastModified'), # Using lastModified as timestamp
        'error_message': get_val('5'),     # Error message
        'stack_frames': parse_stack_frames(get_val('8')), # Stack trace
        'name': entry.get('name')          # Keep name for reference
    }

def normalize_incoming_entry(entry_wrapper: Any) -> Dict[str, Any]:
    """Extracts fields from the Incoming_entry structure into Canonical Schema"""
    # Incoming_entry is a list containing a dict with 'body'
    if isinstance(entry_wrapper, list):
        entry = entry_wrapper[0].get('body', {})
    elif isinstance(entry_wrapper, dict) and 'body' in entry_wrapper:
        entry = entry_wrapper.get('body', {})
    else:
        # Fallback if passed directly as the body dict
        entry = entry_wrapper

    return {
        'entry_id': str(entry.get('id')),
        'project': entry.get('21'),
        'release_stage': entry.get('25'),
        'app_version': entry.get('18'),
        'timestamp': entry.get('lastModified'),
        'error_message': entry.get('5'),
        'stack_frames': parse_stack_frames(entry.get('8')),
        'name': entry.get('name')
    }

# --- 3. HARD GATES ---

def passes_hard_gates(incoming: Dict[str, Any], candidate: Dict[str, Any]) -> tuple[bool, List[str]]:
    reasons = []
    # 1. Self-comparison check
    if incoming['entry_id'] == candidate['entry_id']:
        reasons.append("Self-comparison")
    # 2. Project Mismatch (Critical)
    if incoming['project'] and candidate['project']:
        if incoming['project'] != candidate['project']:
            reasons.append(f"Project mismatch: {incoming['project']} != {candidate['project']}")
    return len(reasons) == 0, reasons

# --- 4. SCORING LOGIC ---

def calculate_stack_score(incoming_frames: List[Dict], candidate_frames: List[Dict]) -> Dict[str, Any]:
    score = 0
    reasons = []
    
    def select_frames(frames):
        non_vendor = [f for f in frames if not f['vendor']]
        vendor = [f for f in frames if f['vendor']]
        chosen = non_vendor if non_vendor else vendor
        unique = []
        seen = set()
        for f in chosen:
            if f['file'] not in seen:
                seen.add(f['file'])
                unique.append(f)
            if len(unique) >= (3 if non_vendor else 2): break
        return unique

    inc_frames = select_frames(incoming_frames)
    cand_frames = select_frames(candidate_frames)
    matched_files = set()
    
    if inc_frames and cand_frames:
        if inc_frames[0]['file'] == cand_frames[0]['file']:
            score += 25
            reasons.append(f"Top frame match: {inc_frames[0]['file']}")
            matched_files.add(inc_frames[0]['file'])
        if len(inc_frames) > 1 and len(cand_frames) > 1:
            if inc_frames[1]['file'] == cand_frames[1]['file']:
                score += 10
                reasons.append(f"Secondary frame match: {inc_frames[1]['file']}")
                matched_files.add(inc_frames[1]['file'])
        for f in inc_frames:
            for cf in cand_frames:
                if f['file'] == cf['file'] and f['file'] not in matched_files:
                    score += 5
                    reasons.append(f"Frame overlap: {f['file']}")
                    matched_files.add(f['file'])

    return {"score": min(score, 40), "reasons": reasons}

def calculate_time_score(incoming_ts: str, candidate_ts: str) -> Dict[str, Any]:
    score = 0
    reasons = []
    if incoming_ts and candidate_ts:
        try:
            dt_inc = pd.to_datetime(incoming_ts)
            dt_cand = pd.to_datetime(candidate_ts)
            days_diff = abs((dt_inc - dt_cand).days)
            if days_diff <= 7:
                score += 5
                reasons.append("Recent (<= 7 days)")
            if days_diff > 180: score -= 15
            elif days_diff > 90: score -= 8
        except:
            pass
    return {"score": score, "reasons": reasons}

def calculate_context_score(incoming: Dict, candidate: Dict) -> Dict[str, Any]:
    score = 0
    reasons = []
    if incoming['project'] and candidate['project'] and incoming['project'] == candidate['project']:
        score += 15
        reasons.append("Same Project")
    if incoming['app_version'] and candidate['app_version'] and incoming['app_version'] == candidate['app_version']:
        score += 10
        reasons.append("Same App Version")
    if incoming['release_stage'] and candidate['release_stage'] and incoming['release_stage'] == candidate['release_stage']:
        score += 5
        reasons.append("Same Release Stage")
    return {"score": score, "reasons": reasons}

def calculate_total_score(incoming: Dict[str, Any], candidate: Dict[str, Any], message_similarity: float = 0.0) -> Dict[str, Any]:
    stack_res = calculate_stack_score(incoming['stack_frames'], candidate['stack_frames'])
    stack_score = stack_res['score']
    
    message_score_scaled = message_similarity * 30
    
    context_res = calculate_context_score(incoming, candidate)
    context_score = context_res['score']
    
    time_res = calculate_time_score(incoming['timestamp'], candidate['timestamp'])
    time_score = time_res['score']
    
    total_score = stack_score + message_score_scaled + context_score + time_score
    final_score = max(0, min(total_score, 100))
    
    all_reasons = (
        stack_res['reasons'] + 
        [f"Message similarity: {message_similarity:.2f}"] + 
        context_res['reasons'] + 
        time_res['reasons']
    )
    
    return {
        "entry_id": candidate['entry_id'],
        "name": candidate['name'],
        "final_score": final_score,
        "scores": {
            "stack": stack_score,
            "message": message_similarity,
            "context": context_score,
            "time": time_score
        },
        "reasons": all_reasons
    }

def generate_triage_report(results: List[Dict[str, Any]], total_candidates: int) -> Dict[str, Any]:
    """
    Generates a report similar to the Bugsnag Triage Agent's output.
    """
    report = {
        "batchSummary": {
            "totalAnalyzed": total_candidates,
            "relatedFound": len(results),
            "confidenceCounts": {"High": 0, "Medium": 0, "Low": 0}
        },
        "relatedEntries": []
    }
    
    for res in results:
        # Map 0-100 score to Confidence
        score = res['final_score']
        if score >= 85: confidence = "High"
        elif score >= 70: confidence = "Medium"
        else: confidence = "Low"
        
        report["batchSummary"]["confidenceCounts"][confidence] += 1
        
        # Generate Explanation from Reasons
        explanation = f"Match found with {confidence} confidence ({score:.1f}/100). "
        explanation += "Key signals: " + "; ".join(res['reasons'][:3]) + "."
        
        entry = {
            "entryId": res['entry_id'],
            "score": score / 100.0, # Normalize to 0-1 for compatibility
            "confidence": confidence,
            "signals": list(res['scores'].keys()),
            "explanation": explanation,
            "breakdown": res['scores']
        }
        report["relatedEntries"].append(entry)
        
    return report

# --- 5. MAIN PIPELINE ---

def run_pipeline(incoming_raw, candidates_raw):
    # Normalize
    incoming_entry = normalize_incoming_entry(incoming_raw)
    candidate_entries = [normalize_veoci_entry(c) for c in candidates_raw]
    
    # Hard Gates & Corpus Building
    eligible_candidates = []
    corpus = [incoming_entry["error_message"] or ""]
    
    for candidate in candidate_entries:
        passed, _ = passes_hard_gates(incoming_entry, candidate)
        if passed:
            corpus.append(candidate["error_message"] or "")
            eligible_candidates.append(candidate)
            
    # TF-IDF
    message_similarities = []
    if len(corpus) > 1:
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_df=0.95)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            incoming_vector = tfidf_matrix[0]
            candidate_vectors = tfidf_matrix[1:]
            message_similarities = cosine_similarity(incoming_vector, candidate_vectors)[0]
        except Exception:
            message_similarities = [0.0] * len(eligible_candidates)
    else:
        message_similarities = []
    
    # Scoring
    results = []
    for i, candidate in enumerate(eligible_candidates):
        raw_msg_sim = message_similarities[i] if i < len(message_similarities) else 0.0
        result = calculate_total_score(incoming_entry, candidate, raw_msg_sim)
        if result['final_score'] > 0:
            results.append(result)
            
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Generate Report
    return generate_triage_report(results, len(eligible_candidates))

if __name__ == "__main__":
    try:
        # Read JSON from stdin
        # Expected format: { "incoming": {...}, "candidates": [...] }
        input_data = json.load(sys.stdin)
        
        incoming = input_data.get("incoming")
        candidates = input_data.get("candidates", [])
        
        if not incoming:
            print(json.dumps({"error": "Missing 'incoming' data"}))
            sys.exit(1)
            
        report = run_pipeline(incoming, candidates)
        
        # Output results
        print(json.dumps(report, indent=2))
        
    except Exception as e:
        # Print error as JSON so n8n can parse it
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
