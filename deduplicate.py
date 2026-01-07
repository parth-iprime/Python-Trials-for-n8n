import sys
import json
import re
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from urllib.parse import urlparse
from difflib import SequenceMatcher
from collections import Counter

# --- 1. HELPER FUNCTIONS ---

def classify_error_mode(error_message: str, error_type: str) -> str:
    """
    Distinguish between INSTANCE bugs and CLASS (infrastructure/systemic) issues.
    """
    msg = (error_message or "").lower()
    et = (error_type or "").lower()

    infra_keywords = [
        # Database/SQL
        'sql connection', 'statementcallback', 'preparedstatementcallback',
        'connection failure', 'aurora', 'jdbc',
        'deadlock', 'too many connections', 'connection reset',
        'resourcefailureexception', 'dataaccessresourcefailure',
        # Redis/Cache
        'redis', 'marking redis', 'redis is down', 'redis command timed out',
        # Monitoring/Alerts
        'monitoring alert', 'recovered from', 'threshold reached',
        'critical threshold', 'warning threshold',
        # Message queues
        'rabbitmq', 'queue size', 'consumer count',
        # Network/Timeouts
        'timeout', 'connection timed out', 'socket timeout',
        'network error', 'connection refused'
    ]

    if any(k in msg for k in infra_keywords):
        return "CLASS"

    if any(k in et for k in ['timeout', 'resource', 'network']):
        return "CLASS"

    return "INSTANCE"

def is_vendor_file(file_path: str) -> bool:
    if not file_path: return False
    vendor_patterns = [
        # JavaScript/Node vendor patterns
        r"^node_modules/", r"^@vue/", r"vuetify/", r"core-js", r"zone\.js",
        r"runtime-core\.esm-bundler\.js", r"reactivity\.esm-bundler\.js",
        r"proxiedModel\.mjs", r"app\..*\.js", r"vendor\..*\.js", r"chunk-.*\.js",
        # Java logging frameworks (with package prefix)
        r"LogbackBugsnagAppender\.java", r"AppenderBase\.java",
        r"AppenderAttachableImpl\.java", r"Logger\.java", r"ch\.qos\.logback\.",
        r"org\.springframework\.", r"org\.apache\.commons\.", r"java\.util\.",
        r"javax?\.", r"sun\.reflect\.",
        # Java Spring Framework files (by filename - catches stack traces without package)
        r"SQLStateSQLExceptionTranslator\.java",
        r"AbstractFallbackSQLExceptionTranslator\.java",
        r"SQLErrorCodeSQLExceptionTranslator\.java",
        r"JdbcTemplate\.java",
        r"TransactionTemplate\.java",
        r"DataSourceTransactionManager\.java",
        r"AbstractPlatformTransactionManager\.java",
        r"TransactionInterceptor\.java",
        r"TransactionAspectSupport\.java",
        # Java servlet/container
        r"DispatcherServlet\.java",
        r"FrameworkServlet\.java",
        r"HttpServlet\.java",
        r"ApplicationFilterChain\.java",
        r"StandardWrapperValve\.java",
        # Java exception handling
        r"ExceptionHandlerExceptionResolver\.java",
        r"ExceptionTranslationFilter\.java",
        # Java thread pool / async
        r"ThreadPoolExecutor\.java",
        r"FutureTask\.java",
        r"CompletableFuture\.java",
    ]
    return any(re.search(p, file_path) for p in vendor_patterns)

def is_generic_frame(file_name: str, line_num: int = None) -> bool:
    """
    Detect generic/minified frames that are too vague for reliable matching.
    These frames indicate bundled code where source maps aren't resolving.
    """
    if not file_name:
        return True
    
    # Remove extension for checking
    base_name = re.sub(r'\.(js|mjs|ts|vue|jsx|tsx)$', '', file_name.lower())
    
    # Generic bundled file patterns
    generic_patterns = [
        r'^app$',           # Generic app bundle
        r'^main$',          # Generic main entry
        r'^index$',         # Generic index
        r'^bundle$',        # Generic bundle
        r'^vendor$',        # Vendor bundle
        r'^chunk',          # Webpack chunks
        r'^runtime',        # Runtime bundles
        r'^polyfills?$',    # Polyfill bundles
        r'^commons?$',      # Common chunks
        r'^shared$',        # Shared bundles
        r'^\d+$',           # Numeric chunk names (e.g., "123.js")
    ]
    
    is_generic = any(re.match(p, base_name) for p in generic_patterns)
    
    # Line 1 is especially suspicious - often indicates unresolved source maps
    if line_num == 1 and is_generic:
        return True
    
    return is_generic

def get_frame_quality(top_frame: str) -> str:
    """
    Assess the quality/specificity of a stack frame for matching.
    Returns: 'high', 'medium', or 'low'
    """
    if not top_frame:
        return 'low'
    
    # Parse "file:line" format
    parts = top_frame.rsplit(':', 1)
    file_name = parts[0] if parts else top_frame
    line_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    
    if is_generic_frame(file_name, line_num):
        return 'low'
    
    # Check for specific component names (Vue, React, etc.)
    specific_patterns = [
        r'[A-Z][a-z]+[A-Z]',  # CamelCase components
        r'\.(vue|component|service|controller|handler)',  # Specific file types
        r'(form|modal|dialog|table|list|view|page)',  # UI components
    ]
    
    if any(re.search(p, file_name, re.IGNORECASE) for p in specific_patterns):
        return 'high'
    
    return 'medium'

# --- STEP 3: FUZZY MATCHING FUNCTIONS ---

def fuzzy_match_score(str1: str, str2: str) -> float:
    """
    Calculate fuzzy match score between two strings (0.0 to 1.0).
    Uses SequenceMatcher for character-level similarity.
    """
    if not str1 or not str2:
        return 0.0
    
    # Normalize strings: lowercase, remove extra whitespace
    s1 = ' '.join(str1.lower().strip().split())
    s2 = ' '.join(str2.lower().strip().split())
    
    if s1 == s2:
        return 1.0
    
    # Use SequenceMatcher for fuzzy matching
    matcher = SequenceMatcher(None, s1, s2)
    return matcher.ratio()

def extract_error_core_message(error_msg: str) -> str:
    """
    Extract the core error message by removing timestamps, URLs, and variable names.
    This allows better fuzzy matching of similar errors.
    """
    if not error_msg:
        return ""
    
    msg = error_msg
    
    # Remove common prefixes (timestamps, dates)
    msg = re.sub(r'^[A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d{1,2}\s+\d{4}\s+[\d:]+\s+GMT[+-]\d{4}\s+\([^)]+\):\s*', '', msg)
    
    # Remove URLs
    msg = re.sub(r'https?://[^\s]+', '[URL]', msg)
    
    # Remove entry/form IDs (common in Veoci errors)
    msg = re.sub(r'\b\d{7,}\b', '[ID]', msg)
    
    # Remove variable-specific names but keep structure
    msg = re.sub(r"'[^']*'", '[VAR]', msg)
    msg = re.sub(r'"[^"]*"', '[VAR]', msg)
    
    return msg.strip()

def categorize_error_type(error_msg: str) -> str:
    """
    Categorize errors into common groups for better matching.
    Returns category name.
    """
    if not error_msg:
        return "Unknown"
    
    msg_lower = error_msg.lower()
    
    # Authentication/Session
    if any(keyword in msg_lower for keyword in ['logged out', 'login', 'auth', 'token', 'session', 'sso', 'firebase']):
        return "Authentication"
    
    # Network/API
    if any(keyword in msg_lower for keyword in ['network error', 'fetch fail', 'timeout', 'api', 'request failed']):
        return "Network"
    
    # Null/Undefined errors
    if any(keyword in msg_lower for keyword in ['null is not', 'undefined is not', 'cannot read propert', 'cannot set propert']):
        return "Null Reference"
    
    # Loading/Resource errors
    if any(keyword in msg_lower for keyword in ['loading chunk', 'failed to load', "didn't start", 'chunk fail']):
        return "Resource Loading"
    
    # UI/Rendering
    if any(keyword in msg_lower for keyword in ['dashboard', 'tiles', 'render', 'leaflet', 'map', 'dialog']):
        return "UI/Rendering"
    
    return "Other"

def extract_error_type(error_message: str) -> str:
    """Extract error type from error message (e.g., TypeError, Error, ReferenceError)"""
    if not error_message:
        return "Unknown"
    # Common JS error types
    error_patterns = [
        r'^(TypeError|ReferenceError|SyntaxError|RangeError|URIError|EvalError|Error):',
        r'^(TypeError|ReferenceError|SyntaxError|RangeError|URIError|EvalError|Error)\b',
        r'(TypeError|ReferenceError|SyntaxError|RangeError|URIError|EvalError):'
    ]
    for pattern in error_patterns:
        match = re.search(pattern, error_message, re.IGNORECASE)
        if match:
            return match.group(1)
    # Check for common error indicators
    if 'undefined is not' in error_message or 'null is not' in error_message:
        return "TypeError"
    if 'is not defined' in error_message:
        return "ReferenceError"
    return "Error"

def extract_route(url_or_path: str) -> str:
    """Extract route/path from URL or path string. Handles 'GET /api/...' format."""
    if not url_or_path:
        return None  # Return None instead of "/" so we know it's missing
    
    # Strip HTTP method prefix if present (e.g., "GET /api/v2/me/keepAlive")
    route_str = url_or_path.strip()
    method_match = re.match(r'^(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\s+(.+)$', route_str, re.IGNORECASE)
    if method_match:
        route_str = method_match.group(2)
    
    try:
        parsed = urlparse(route_str)
        path = parsed.path or "/"
        # Remove common prefixes and clean up
        path = re.sub(r'^/v/c/\d+', '', path)  # Veoci-specific cleanup
        return path if path else "/"
    except:
        return "/"

def parse_browser_info(browser: str, ua_string: str) -> Dict[str, Any]:
    """Parse browser/client information"""
    result = {
        'browser_family': None,
        'browser_version': None,
        'is_mobile': False
    }
    
    if browser:
        # Parse browser like "Mobile Safari UI/WKWebView 18.2"
        match = re.match(r'^(.+?)\s+([\d.]+)?$', browser)
        if match:
            result['browser_family'] = match.group(1).strip()
            result['browser_version'] = match.group(2)
        else:
            result['browser_family'] = browser
    
    # Check for mobile indicators
    mobile_indicators = ['Mobile', 'Safari UI', 'WKWebView', 'iOS', 'Android']
    if browser and any(ind in browser for ind in mobile_indicators):
        result['is_mobile'] = True
    if ua_string and any(ind in ua_string for ind in mobile_indicators):
        result['is_mobile'] = True
        
    return result

def normalize_environment(release_stage: str, app_type: str) -> str:
    """Normalize environment string (e.g., PROD-MOBILE)"""
    env_parts = []
    
    if release_stage:
        stage = release_stage.upper()
        if 'PROD' in stage or 'PRODUCTION' in stage:
            env_parts.append('PROD')
        elif 'STAG' in stage:
            env_parts.append('STAGING')
        elif 'DEV' in stage:
            env_parts.append('DEV')
        else:
            env_parts.append(stage)
    
    if app_type:
        if 'mobile' in app_type.lower():
            env_parts.append('MOBILE')
        elif 'client' in app_type.lower():
            env_parts.append('WEB')
        elif 'api' in app_type.lower():
            env_parts.append('API')
            
    return '-'.join(env_parts) if env_parts else None

# --- TEXT SIMILARITY (TF-IDF) ---

def calculate_semantic_similarities(incoming_text: str, candidate_texts: List[str]) -> List[float]:
    """
    Calculate text similarity between incoming error and all candidates using TF-IDF.
    """
    if not candidate_texts:
        return []
    
    corpus = [incoming_text or ""] + [t or "" for t in candidate_texts]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_df=0.95)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        incoming_vector = tfidf_matrix[0]
        candidate_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(incoming_vector, candidate_vectors)[0]
        return list(similarities)
    except Exception:
        return [0.0] * len(candidate_texts)

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
        if isinstance(val, list): return ", ".join(str(v) for v in val) # Handle multi-value
        return val

    error_message = get_val('5')
    release_stage = get_val('25')
    app_type = get_val('26')
    browser = get_val('16')
    ua_string = get_val('13')
    request_url = get_val('39')
    error_url = get_val('6')  # Route like "GET /api/v2/me/keepAlive"
    stack_trace = get_val('8')
    
    # Parse browser info
    browser_info = parse_browser_info(browser, ua_string)
    
    # Get top frame from stack
    stack_frames = parse_stack_frames(stack_trace)
    top_frame = None
    if stack_frames:
        tf = stack_frames[0]
        top_frame = f"{tf['file'].replace('.js', '')}:{tf['line']}" if tf.get('line') else tf['file'].replace('.js', '')
    
    # Calculate stack overlap info
    stack_files = [f['file'] for f in stack_frames if not f.get('vendor')][:5]

    return {
        'entry_id': str(entry.get('id')),
        'project': get_val('21'),          # Bugsnag Project ID
        'release_stage': release_stage,    # Release Stage
        'app_version': get_val('18'),      # App Version
        'app_type': app_type,              # App Type (client type)
        'timestamp': entry.get('lastModified'),
        'error_message': error_message,    # Error message
        'error_type': extract_error_type(error_message),  # Error type
        'stack_frames': stack_frames,      # Full stack trace
        'stack_files': stack_files,        # Non-vendor file names for comparison
        'top_frame': top_frame,            # Top frame string
        'name': entry.get('name'),
        # New fields
        'browser': browser,
        'browser_family': browser_info['browser_family'],
        'browser_version': browser_info['browser_version'],
        'is_mobile': browser_info['is_mobile'],
        'os_name': get_val('15'),
        'request_url': request_url,
        # Use field 6 (error_url like "GET /api/...") if request_url (field 39) is empty
        'route': extract_route(request_url) or extract_route(error_url),
        'environment': normalize_environment(release_stage, app_type),
        'linked_tickets': get_val('24'),   # Linked Veoci Tickets
        'bugsnag_url': get_val('0'),       # Bugsnag URL
        'error_url': error_url             # Error URL (route with method)
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

    error_message = entry.get('5')
    release_stage = entry.get('25')
    app_type = entry.get('26')
    browser = entry.get('16')
    ua_string = entry.get('13')
    request_url = entry.get('39')
    stack_trace = entry.get('8')
    
    # Parse browser info
    browser_info = parse_browser_info(browser, ua_string)
    
    # Get top frame from stack
    stack_frames = parse_stack_frames(stack_trace)
    top_frame = None
    if stack_frames:
        tf = stack_frames[0]
        top_frame = f"{tf['file'].replace('.js', '')}:{tf['line']}" if tf.get('line') else tf['file'].replace('.js', '')
    
    # Calculate stack overlap info
    stack_files = [f['file'] for f in stack_frames if not f.get('vendor')][:5]

    return {
        'entry_id': str(entry.get('id')),
        'project': entry.get('21'),
        'release_stage': release_stage,
        'app_version': entry.get('18'),
        'app_type': app_type,
        'timestamp': entry.get('lastModified'),
        'error_message': error_message,
        'error_type': extract_error_type(error_message),
        'stack_frames': stack_frames,
        'stack_files': stack_files,
        'top_frame': top_frame,
        'name': entry.get('name'),
        # New fields
        'browser': browser,
        'browser_family': browser_info['browser_family'],
        'browser_version': browser_info['browser_version'],
        'is_mobile': browser_info['is_mobile'],
        'os_name': entry.get('15'),
        'request_url': request_url,
        # Use field 6 (error_url like "GET /api/...") if request_url (field 39) is empty
        'route': extract_route(request_url) or extract_route(entry.get('6')),
        'environment': normalize_environment(release_stage, app_type),
        'linked_tickets': entry.get('24'),
        'bugsnag_url': entry.get('0'),
        'error_url': entry.get('6')
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

def calculate_stack_score(incoming: Dict, candidate: Dict) -> Dict[str, Any]:
    """
    Enhanced stack trace comparison with overlap percentage.
    Applies penalties for generic/minified frames.
    Throttles score for CLASS (infrastructure) errors.
    """
    score = 0
    reasons = []
    signals = []
    stack_overlap = 0
    frame_quality = 'medium'
    
    # Read error mode for CLASS throttling
    error_mode = incoming.get('_error_mode', 'INSTANCE')
    
    incoming_frames = incoming.get('stack_frames', [])
    candidate_frames = candidate.get('stack_frames', [])
    
    def select_frames(frames):
        non_vendor = [f for f in frames if not f.get('vendor')]
        vendor = [f for f in frames if f.get('vendor')]
        chosen = non_vendor if non_vendor else vendor
        unique = []
        seen = set()
        for f in chosen:
            if f['file'] not in seen:
                seen.add(f['file'])
                unique.append(f)
            if len(unique) >= 5: break
        return unique, len(non_vendor) == 0 and len(vendor) > 0  # Flag if vendor-only

    inc_frames, inc_vendor_only = select_frames(incoming_frames)
    cand_frames, cand_vendor_only = select_frames(candidate_frames)
    
    # CRITICAL: Vendor-only stacks (all logging/framework frames) are NOT meaningful
    # They indicate infrastructure logging, not actual application code
    is_vendor_only_stack = inc_vendor_only or cand_vendor_only
    
    # Check if top frame is generic/minified
    top_frame_str = incoming.get('top_frame', '')
    frame_quality = get_frame_quality(top_frame_str)
    is_generic = frame_quality == 'low' or is_vendor_only_stack
    
    # Calculate overlap percentage
    if inc_frames and cand_frames:
        inc_files = set(f['file'] for f in inc_frames)
        cand_files = set(f['file'] for f in cand_frames)
        common = inc_files.intersection(cand_files)
        if inc_files or cand_files:
            stack_overlap = round(len(common) / max(len(inc_files), len(cand_files)) * 100, 1)
    
    matched_files = set()
    
    # Top frame match (most important - but penalize generic frames)
    if inc_frames and cand_frames:
        if inc_frames[0]['file'] == cand_frames[0]['file']:
            top_file = inc_frames[0]['file'].replace('.js', '')
            
            if is_generic:
                # Generic frames get minimal score - they're unreliable
                score += 5  # Was 20
                reasons.append(f"Top frame ({top_file}) matches (⚠️ generic/minified)")
                # Don't add 'top_frame' signal for generic frames
            else:
                score += 20
                reasons.append(f"Top frame ({top_file}) matches")
                signals.append('top_frame')
            
            matched_files.add(inc_frames[0]['file'])
            
            # Line number match bonus - less valuable for generic frames
            if inc_frames[0].get('line') and cand_frames[0].get('line'):
                if inc_frames[0]['line'] == cand_frames[0]['line']:
                    if is_generic and inc_frames[0]['line'] == 1:
                        # Line 1 on generic bundle = worthless
                        score += 0
                        reasons.append("Line 1 match (ignored - likely minified)")
                    elif is_generic:
                        score += 2  # Reduced bonus
                        reasons.append("Exact line number match (generic frame)")
                    else:
                        score += 5
                        reasons.append("Exact line number match")
        
        # Secondary frame match
        if len(inc_frames) > 1 and len(cand_frames) > 1:
            if inc_frames[1]['file'] == cand_frames[1]['file']:
                secondary_generic = is_generic_frame(inc_frames[1]['file'], inc_frames[1].get('line'))
                if secondary_generic:
                    score += 3  # Reduced from 10
                    reasons.append(f"Secondary frame match: {inc_frames[1]['file']} (generic)")
                else:
                    score += 10
                    reasons.append(f"Secondary frame match: {inc_frames[1]['file']}")
                matched_files.add(inc_frames[1]['file'])
        
        # Stack overlap bonus - reduced for generic stacks
        if stack_overlap >= 80:
            if is_generic:
                score += 5  # Was 15
                reasons.append(f"High stack similarity ({stack_overlap}%) (⚠️ generic stack)")
            else:
                score += 15
                signals.append('stack')
                reasons.append(f"High stack similarity ({stack_overlap}%)")
        elif stack_overlap >= 50:
            if is_generic:
                score += 3  # Was 8
                reasons.append(f"Moderate stack similarity ({stack_overlap}%) (generic)")
            else:
                score += 8
                signals.append('stack')
                reasons.append(f"Moderate stack similarity ({stack_overlap}%)")
        elif stack_overlap > 0:
            score += 2 if is_generic else 3
            reasons.append(f"Some stack overlap ({stack_overlap}%)")

    # Add warning for vendor-only stacks
    if is_vendor_only_stack:
        reasons.append("⚠️ Vendor-only stack (logging framework) - low reliability")
        frame_quality = 'vendor-only'

    # Cap stack score based on error mode (CLASS errors get throttled)
    # Vendor-only stacks also get heavily throttled
    if error_mode == "INSTANCE" and not is_vendor_only_stack:
        max_stack = 50
    elif is_vendor_only_stack:
        max_stack = 10  # Vendor-only stacks are nearly meaningless
    else:
        max_stack = 20
    
    return {
        "score": min(score, max_stack),
        "reasons": reasons,
        "signals": signals,
        "stack_overlap": stack_overlap,
        "top_frame": incoming.get('top_frame'),
        "frame_quality": frame_quality
    }

def calculate_message_score(message_similarity: float, incoming: Dict = None) -> Dict[str, Any]:
    """Score based on error message similarity. Penalizes SQL dynamic query differences."""
    signals = []
    reasons = []
    
    # Check for SQL-related messages
    msg_lower = (incoming.get('error_message') or "").lower() if incoming else ""
    is_sql = 'sql' in msg_lower or 'statementcallback' in msg_lower
    
    # Scale TF-IDF similarity (0-1) to score contribution
    if message_similarity >= 0.9:
        score = 20
        reasons.append("Identical normalized message")
        signals.append('message')
    elif message_similarity >= 0.7:
        score = 15
        reasons.append(f"Message highly similar ({message_similarity:.0%})")
        signals.append('message')
    elif message_similarity >= 0.4:
        score = 8
        reasons.append(f"Message moderately similar ({message_similarity:.0%})")
        signals.append('message')
    else:
        score = message_similarity * 10  # 0-4 points
        if message_similarity > 0.1:
            reasons.append(f"Message slightly similar ({message_similarity:.0%})")
    
    # SQL message entropy penalty - dynamic queries differ
    if is_sql and message_similarity < 0.6:
        score = score * 0.5
        reasons.append("SQL message differs (dynamic query)")
    
    return {"score": score, "reasons": reasons, "signals": signals}

def calculate_environment_score(incoming: Dict, candidate: Dict) -> Dict[str, Any]:
    """Score based on environment, client, and route matching"""
    score = 0
    reasons = []
    signals = []
    
    # Environment match (e.g., PROD-MOBILE)
    if incoming.get('environment') and candidate.get('environment'):
        if incoming['environment'] == candidate['environment']:
            score += 10
            reasons.append(f"Same environment ({incoming['environment']})")
            signals.append('env')
    
    # Route/URL match
    if incoming.get('route') and candidate.get('route'):
        if incoming['route'] == candidate['route'] and incoming['route'] != '/':
            score += 8
            reasons.append(f"Exact route match ({incoming['route']})")
            signals.append('url')
        elif incoming['route'] != '/' and candidate['route'] != '/':
            # Partial route match
            inc_parts = incoming['route'].strip('/').split('/')
            cand_parts = candidate['route'].strip('/').split('/')
            if inc_parts and cand_parts and inc_parts[0] == cand_parts[0]:
                score += 3
                reasons.append("Route prefix match")
    
    # Client/Browser match - but "undefined" should not count as a match!
    inc_browser = incoming.get('browser_family', '')
    cand_browser = candidate.get('browser_family', '')
    # Exclude meaningless client values
    meaningless_clients = ['undefined', 'unknown', '', None, 'null', 'none']
    if (inc_browser and cand_browser and 
        inc_browser.lower() not in meaningless_clients and
        cand_browser.lower() not in meaningless_clients):
        if inc_browser == cand_browser:
            if incoming.get('browser_version') == candidate.get('browser_version'):
                score += 5
                reasons.append(f"Same client family and major version ({inc_browser} {incoming.get('browser_version')})")
            else:
                score += 3
                reasons.append(f"Same client family ({inc_browser}), different version")
            signals.append('client')
    
    # App type match
    if incoming.get('app_type') and candidate.get('app_type'):
        if incoming['app_type'] == candidate['app_type']:
            score += 3
            reasons.append(f"Same app type ({incoming['app_type']})")
    
    return {"score": min(score, 25), "reasons": reasons, "signals": signals}

def calculate_temporal_score(incoming: Dict, candidate: Dict) -> Dict[str, Any]:
    """Score based on time proximity"""
    score = 0
    reasons = []
    signals = []
    
    incoming_ts = incoming.get('timestamp')
    candidate_ts = candidate.get('timestamp')
    
    if incoming_ts and candidate_ts:
        try:
            dt_inc = pd.to_datetime(incoming_ts, unit='ms')
            dt_cand = pd.to_datetime(candidate_ts, unit='ms')
            hours_diff = abs((dt_inc - dt_cand).total_seconds() / 3600)
            days_diff = hours_diff / 24
            
            if hours_diff <= 72:  # Within 72 hours
                score += 5
                reasons.append("Temporal proximity: within 72 hours")
                signals.append('temporal')
            elif days_diff <= 7:
                score += 3
                reasons.append("Temporal proximity: within 7 days")
                signals.append('temporal')
            elif days_diff > 180:
                score -= 10
                reasons.append("Temporal distance: over 180 days")
            elif days_diff > 90:
                score -= 5
                reasons.append("Temporal proximity: over 90 days")
        except:
            pass
    
    return {"score": score, "reasons": reasons, "signals": signals}

def calculate_total_score(incoming: Dict[str, Any], candidate: Dict[str, Any], message_similarity: float = 0.0) -> Dict[str, Any]:
    """
    Calculate total similarity score with all signals.
    Max score: 100 (stack: 50, message: 20, env: 25, temporal: 5)
    """
    # Calculate individual scores
    stack_res = calculate_stack_score(incoming, candidate)
    message_res = calculate_message_score(message_similarity, incoming)
    env_res = calculate_environment_score(incoming, candidate)
    temporal_res = calculate_temporal_score(incoming, candidate)
    
    # Sum scores
    total_score = (
        stack_res['score'] + 
        message_res['score'] + 
        env_res['score'] + 
        temporal_res['score']
    )
    final_score = max(0, min(total_score, 100))
    
    # Collect all signals and reasons
    all_signals = (
        stack_res.get('signals', []) + 
        message_res.get('signals', []) + 
        env_res.get('signals', []) + 
        temporal_res.get('signals', [])
    )
    
    all_reasons = (
        stack_res['reasons'] + 
        message_res['reasons'] + 
        env_res['reasons'] + 
        temporal_res['reasons']
    )
    
    return {
        "entry_id": candidate['entry_id'],
        "name": candidate['name'],
        "final_score": final_score,
        "signals": list(set(all_signals)),  # Unique signals
        "scores": {
            "stack": stack_res['score'],
            "message": round(message_similarity, 2),
            "environment": env_res['score'],
            "temporal": temporal_res['score']
        },
        "reasons": all_reasons,
        # Additional metadata for display
        "top_frame": candidate.get('top_frame'),
        "stack_overlap": stack_res.get('stack_overlap', 0),
        "error_type": candidate.get('error_type'),
        "route": candidate.get('route'),
        "environment": candidate.get('environment'),
        "linked_tickets": candidate.get('linked_tickets'),
        "error_mode": incoming.get('_error_mode', 'INSTANCE')
    }

def generate_triage_report(results: List[Dict[str, Any]], total_candidates: int, min_score: int = 0) -> Dict[str, Any]:
    """
    Generates a report similar to the Bugsnag Triage Agent's output.
    Includes all metadata for rich HTML rendering.
    
    Confidence thresholds (adjusted for better match quality):
    - High: score >= 70 (was 85)
    - Medium: score >= 50 (was 70)  
    - Low: score >= 30
    - Below 30: filtered out by default
    """
    # WEAK SIGNALS that alone are NOT enough to qualify a match
    # These can boost a score but shouldn't be the only signal
    WEAK_SIGNALS = {'client', 'temporal'}
    MEANINGFUL_SIGNALS = {'stack', 'top_frame', 'message', 'env', 'url'}
    
    # Filter results by minimum score threshold AND signal quality
    filtered_results = []
    for r in results:
        if r['final_score'] < min_score:
            continue
        
        signals = set(r.get('signals', []))
        # Require at least one MEANINGFUL signal (not just client/temporal)
        has_meaningful_signal = bool(signals & MEANINGFUL_SIGNALS)
        
        if has_meaningful_signal:
            filtered_results.append(r)
        # else: skip entries with only weak signals like "client"
    
    report = {
        "batchSummary": {
            "totalAnalyzed": total_candidates,
            "relatedFound": len(filtered_results),
            "confidenceCounts": {"High": 0, "Medium": 0, "Low": 0}
        },
        "relatedEntries": []
    }
    
    for res in filtered_results:
        score = res['final_score']
        error_mode = res.get('error_mode', 'INSTANCE')
        
        # CLASS mode can NEVER be High confidence
        if error_mode == "CLASS":
            confidence = "Low"  # Infrastructure errors always Low
        else:
            # Adjusted confidence thresholds for INSTANCE errors
            if score >= 70:
                confidence = "High"
            elif score >= 50:
                confidence = "Medium"
            else:
                confidence = "Low"
        
        report["batchSummary"]["confidenceCounts"][confidence] += 1
        
        # Generate detailed explanation from reasons
        reasons_str = "; ".join(res['reasons'][:4]) if res['reasons'] else "General similarity detected"
        explanation = f"{reasons_str}."
        
        entry = {
            "entryId": res['entry_id'],
            "name": res.get('name'),
            "score": round(score / 100.0, 2),  # Normalize to 0-1
            "confidence": confidence,
            "signals": res.get('signals', []),
            "explanation": explanation,
            "breakdown": res['scores'],
            # Rich metadata for display
            "topFrame": res.get('top_frame'),
            "stackOverlap": res.get('stack_overlap', 0),
            "errorType": res.get('error_type'),
            "route": res.get('route'),
            "environment": res.get('environment'),
            "linkedTickets": res.get('linked_tickets'),
            "errorMode": res.get('error_mode')
        }
        report["relatedEntries"].append(entry)
    
    # HARD CAP: Limit to top 10 matches (non-negotiable)
    MAX_RELATED = 10
    report["relatedEntries"] = report["relatedEntries"][:MAX_RELATED]
    
    # Update relatedFound to reflect ACTUAL capped count shown to user
    report["batchSummary"]["relatedFound"] = len(report["relatedEntries"])
    
    # RECALCULATE confidence counts based on CAPPED entries only
    report["batchSummary"]["confidenceCounts"] = {"High": 0, "Medium": 0, "Low": 0}
    for entry in report["relatedEntries"]:
        report["batchSummary"]["confidenceCounts"][entry["confidence"]] += 1
    
    return report

# --- STEP 4: TREND ANALYSIS ---

def analyze_frequency_trends(incoming: Dict, candidates: List[Dict], related_entries: List[Dict]) -> Dict[str, Any]:
    """
    Analyze frequency and trends of similar errors over time.
    Returns insights about error patterns, spikes, and affected users.
    """
    
    # Extract incoming error details
    inc_msg_core = extract_error_core_message(incoming.get('error_message', ''))
    inc_category = categorize_error_type(incoming.get('error_message', ''))
    
    # Filter candidates to similar errors (same category or similar message)
    similar_errors = []
    for cand in candidates:
        cand_msg = cand.get('error_message', '')
        cand_msg_core = extract_error_core_message(cand_msg)
        cand_category = categorize_error_type(cand_msg)
        
        # Check if similar
        if cand_category == inc_category or fuzzy_match_score(inc_msg_core, cand_msg_core) > 0.6:
            similar_errors.append(cand)
    
    # Time-based analysis
    now = datetime.utcnow()
    time_buckets = {
        'last_24h': [],
        'last_7d': [],
        'last_30d': []
    }
    
    for error in similar_errors:
        try:
            # Parse timestamp (milliseconds)
            timestamp_val = error.get('timestamp')
            if timestamp_val:
                if isinstance(timestamp_val, str):
                    if timestamp_val.endswith('Z'):
                        error_time = datetime.fromisoformat(timestamp_val[:-1])
                    else:
                        error_time = datetime.fromisoformat(timestamp_val)
                else:
                    # Assume milliseconds timestamp
                    error_time = datetime.utcfromtimestamp(int(timestamp_val) / 1000)
                
                time_diff = now - error_time
                
                if time_diff <= timedelta(hours=24):
                    time_buckets['last_24h'].append(error)
                if time_diff <= timedelta(days=7):
                    time_buckets['last_7d'].append(error)
                if time_diff <= timedelta(days=30):
                    time_buckets['last_30d'].append(error)
        except:
            pass
    
    # Daily breakdown for last 7 days
    daily_counts = {}
    for error in time_buckets['last_7d']:
        try:
            timestamp_val = error.get('timestamp')
            if timestamp_val:
                if isinstance(timestamp_val, str):
                    if timestamp_val.endswith('Z'):
                        error_time = datetime.fromisoformat(timestamp_val[:-1])
                    else:
                        error_time = datetime.fromisoformat(timestamp_val)
                else:
                    error_time = datetime.utcfromtimestamp(int(timestamp_val) / 1000)
                
                date_key = error_time.strftime('%b %d')
                daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
        except:
            pass
    
    # Detect spike (compare last 24h to previous days)
    avg_daily_last_week = len(time_buckets['last_7d']) / 7 if time_buckets['last_7d'] else 0
    count_last_24h = len(time_buckets['last_24h'])
    
    is_spike = count_last_24h > (avg_daily_last_week * 2) and count_last_24h > 3
    spike_percentage = ((count_last_24h / avg_daily_last_week) - 1) * 100 if avg_daily_last_week > 0 else 0
    
    # Affected users (extract from error messages or user fields)
    affected_users = set()
    for error in similar_errors:
        msg = error.get('error_message', '')
        # Extract email patterns
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', msg)
        affected_users.update(emails)
    
    # Environment breakdown
    env_counts = Counter()
    for error in similar_errors:
        env = error.get('environment', 'Unknown')
        if env:
            env_counts[env] += 1
    
    return {
        'total_similar': len(similar_errors),
        'counts': {
            'last_24h': len(time_buckets['last_24h']),
            'last_7d': len(time_buckets['last_7d']),
            'last_30d': len(time_buckets['last_30d'])
        },
        'daily_breakdown': dict(sorted(daily_counts.items())),
        'spike_detected': is_spike,
        'spike_percentage': round(spike_percentage, 1) if spike_percentage else 0,
        'affected_users': list(affected_users)[:10],  # Top 10
        'affected_users_count': len(affected_users),
        'environment_breakdown': dict(env_counts.most_common(5)),
        'category': inc_category
    }

# --- 5. MAIN PIPELINE ---

def run_pipeline(incoming_raw, candidates_raw):
    # Normalize incoming
    incoming_entry = normalize_incoming_entry(incoming_raw)
    
    # Extract candidate entries from n8n format if needed
    # n8n format: [{'json': {'entries': [...], ...}, 'pairedItem': ...}]
    # Direct format: [entry1, entry2, ...]
    if candidates_raw and isinstance(candidates_raw, list):
        first_item = candidates_raw[0] if candidates_raw else {}
        if isinstance(first_item, dict) and 'json' in first_item:
            # n8n format - extract entries from nested structure
            entries_list = first_item.get('json', {}).get('entries', [])
            candidate_entries = [normalize_veoci_entry(c) for c in entries_list]
        else:
            # Direct format - entries are passed directly
            candidate_entries = [normalize_veoci_entry(c) for c in candidates_raw]
    else:
        candidate_entries = []
    
    # CHANGE 1: Classify error mode (INSTANCE vs CLASS)
    error_mode = classify_error_mode(
        incoming_entry.get('error_message'),
        incoming_entry.get('error_type')
    )
    incoming_entry['_error_mode'] = error_mode
    
    # Detect if incoming has generic stack frame
    incoming_frame_quality = get_frame_quality(incoming_entry.get('top_frame', ''))
    has_generic_stack = incoming_frame_quality == 'low'
    
    # Hard Gates & Corpus Building
    eligible_candidates = []
    candidate_messages = []
    
    for candidate in candidate_entries:
        passed, _ = passes_hard_gates(incoming_entry, candidate)
        if passed:
            # Additional soft gate: when stack is generic, prefer same error_type
            # (don't hard-block, but we'll use this info in scoring)
            candidate['_error_type_match'] = (
                incoming_entry.get('error_type') == candidate.get('error_type')
            )
            candidate_messages.append(candidate["error_message"] or "")
            eligible_candidates.append(candidate)
    
    # Calculate message similarities (semantic or TF-IDF)
    incoming_message = incoming_entry["error_message"] or ""
    
    if eligible_candidates:
        message_similarities = calculate_semantic_similarities(
            incoming_message, 
            candidate_messages
        )
    else:
        message_similarities = []
    
    # Scoring
    results = []
    for i, candidate in enumerate(eligible_candidates):
        raw_msg_sim = message_similarities[i] if i < len(message_similarities) else 0.0
        
        # When stack is generic, require higher message similarity for high scores
        if has_generic_stack and raw_msg_sim < 0.5 and not candidate.get('_error_type_match', False):
            # Apply a penalty - generic stack + weak message = unreliable match
            raw_msg_sim = raw_msg_sim * 0.5  # Halve the contribution
        
        result = calculate_total_score(incoming_entry, candidate, raw_msg_sim)
        
        # Propagate error_mode into results
        result['error_mode'] = error_mode
        
        # Add error type match bonus/penalty
        if has_generic_stack:
            if candidate.get('_error_type_match', False):
                result['final_score'] = min(100, result['final_score'] + 5)
                result['reasons'].append(f"Error type match ({incoming_entry.get('error_type')})")
            else:
                result['final_score'] = max(0, result['final_score'] - 10)
                result['reasons'].append(f"Error type mismatch: {incoming_entry.get('error_type')} vs {candidate.get('error_type')}")
        
        if result['final_score'] > 0:
            results.append(result)
            
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Generate Report
    report = generate_triage_report(results, len(eligible_candidates))
    
    # Add trend analysis
    related_entries = report.get('relatedEntries', [])
    trend_analysis = analyze_frequency_trends(incoming_entry, candidate_entries, related_entries)
    report['trendAnalysis'] = trend_analysis
    
    # Add metadata about matching method
    report['metadata'] = {
        'similarity_method': 'tfidf',
        'incoming_frame_quality': incoming_frame_quality,
        'generic_stack_detected': has_generic_stack
    }
    
    return report

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
