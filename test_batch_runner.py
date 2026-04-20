import os
from src.memfaith.schemas import NormalizedExample, SourceSegment
from src.memfaith.backends import HeuristicBackend
from src.memfaith.batch_runner import BatchCCSRunner
from src.memfaith.cache import SQLitePredictionCache

if os.path.exists("test_cache.db"):
    os.remove("test_cache.db")
if os.path.exists("test_out.jsonl"):
    os.remove("test_out.jsonl")

example = NormalizedExample(
    dataset="test",
    example_id="1",
    query="Is this a test?",
    gold_answer="SUPPORTS",
    task_type="classification",
    evidence_segments=[SourceSegment(segment_id=0, title="t1", text="This is a test evidence.", is_gold=True, source_type="ev")],
    distractor_segments=[SourceSegment(segment_id=1, title="t2", text="This is distractor 1.", is_gold=False, source_type="dis")],
    metadata={"required_segment_ids": [0]}
)

cache = SQLitePredictionCache("test_cache.db")
backend = HeuristicBackend()
runner = BatchCCSRunner(backend=backend, cache=cache)

print("Running batch runner for k_values=[2]")
records = runner.run([example], k_values=[2], output_path="test_out.jsonl")
print("CCS Flip rate:", records[0].get("ccs_example"))
print("Done. Check test_out.jsonl and test_cache.db")
