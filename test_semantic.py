import json
import sys
sys.path.insert(0, '.')
from deduplicate import run_pipeline

# Simulated incoming error
incoming = [{
    'body': {
        'id': 'new-error',
        '5': "TypeError: Cannot read property 'address' of undefined",
        '8': '<strong>app.js:1</strong> - anonymous<br><strong>chart.mjs:50</strong>',
        '21': 'project-123',
        '25': 'production',
        '26': 'mobile-client',
        '16': 'Mobile Safari UI/WKWebView 18.2',
        '39': 'https://app.example.com/'
    }
}]

# Simulated candidates with different error messages
candidates = [
    {
        'id': '1001',
        'name': 'Similar Error 1 - Same property access issue',
        'values': {
            '5': {'data': {'value': "TypeError: undefined is not an object (evaluating 't.address')"}},
            '8': {'data': {'value': '<strong>app.js:1</strong><br><strong>chart.mjs:50</strong>'}},
            '21': {'data': {'value': 'project-123'}},
            '25': {'data': {'value': 'production'}},
            '26': {'data': {'value': 'mobile-client'}},
            '16': {'data': {'value': 'Mobile Safari UI/WKWebView 18.2'}},
            '39': {'data': {'value': 'https://app.example.com/'}}
        }
    },
    {
        'id': '1002', 
        'name': 'Different Error - Barcode regex issue',
        'values': {
            '5': {'data': {'value': "null is not an object (evaluating 'this.barCodeValue.match')"}},
            '8': {'data': {'value': '<strong>app.js:1</strong><br><strong>chart.mjs:50</strong>'}},
            '21': {'data': {'value': 'project-123'}},
            '25': {'data': {'value': 'production'}},
            '26': {'data': {'value': 'mobile-client'}},
            '16': {'data': {'value': 'Mobile Safari UI/WKWebView 18.2'}},
            '39': {'data': {'value': 'https://app.example.com/'}}
        }
    },
    {
        'id': '1003',
        'name': 'Completely Different - Readonly property',
        'values': {
            '5': {'data': {'value': 'Attempted to assign to readonly property'}},
            '8': {'data': {'value': '<strong>app.js:1</strong><br><strong>chart.mjs:50</strong>'}},
            '21': {'data': {'value': 'project-123'}},
            '25': {'data': {'value': 'production'}},
            '26': {'data': {'value': 'mobile-client'}},
            '16': {'data': {'value': 'Mobile Safari UI/WKWebView 18.2'}},
            '39': {'data': {'value': 'https://app.example.com/'}}
        }
    },
    {
        'id': '1004',
        'name': 'Very Similar - Same concept different words',
        'values': {
            '5': {'data': {'value': "Cannot access property 'address' on undefined value"}},
            '8': {'data': {'value': '<strong>app.js:1</strong><br><strong>chart.mjs:50</strong>'}},
            '21': {'data': {'value': 'project-123'}},
            '25': {'data': {'value': 'production'}},
            '26': {'data': {'value': 'mobile-client'}},
            '16': {'data': {'value': 'Mobile Safari UI/WKWebView 18.2'}},
            '39': {'data': {'value': 'https://app.example.com/'}}
        }
    }
]

print("=" * 60)
print("INCOMING ERROR:")
print(f"  Message: {incoming[0]['body']['5']}")
print(f"  Stack: app.js:1 (GENERIC)")
print("=" * 60)
print()

result = run_pipeline(incoming, candidates)

print()
print("=" * 60)
print("RESULTS (sorted by score):")
print("=" * 60)

for entry in result['relatedEntries']:
    print(f"\n{entry['name']}")
    print(f"  Score: {entry['score']:.2f} ({entry['confidence']})")
    print(f"  Message similarity: {entry['breakdown']['message']:.2f}")
    print(f"  Signals: {', '.join(entry['signals']) if entry['signals'] else 'none'}")
    print(f"  Explanation: {entry['explanation'][:100]}...")

print()
print("=" * 60)
print("METADATA:")
print(f"  Method: {result.get('metadata', {}).get('similarity_method', 'unknown')}")
print(f"  Generic stack detected: {result.get('metadata', {}).get('generic_stack_detected', 'unknown')}")
print("=" * 60)
