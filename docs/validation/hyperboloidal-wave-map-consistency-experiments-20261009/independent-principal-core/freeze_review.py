from pathlib import Path
import hashlib,json,shutil,subprocess
P=Path(__file__).resolve().parent;F=P/'immutable-independent-principal-core-review-20261009';assert not F.exists()
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest()
items=[q for q in sorted(P.rglob('*'))if q.is_file()and F not in q.parents and'__pycache__'not in q.parts]
for q in items:
 if q.suffix=='.json':json.dumps(json.loads(q.read_text()),allow_nan=False)
F.mkdir();records=[]
for q in items:
 r={'path':str(q.relative_to(P)),'original_path':str(q),'sha256':sha(q),'bytes':q.stat().st_size};d=F/r['path'];d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,d);assert sha(d)==r['sha256'];records.append(r)
obj={'kind':'independent exact Fraction and saved-only principal/core review','passed':True,'capture_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'capsule_index_sha256':'569ebb305b36a41fdb091330fce7da8e5be7a50f658588ca3892a51d68274ced','files':records,'file_count':len(records),'bytes':sum(q['bytes']for q in records),'receipt_sha256':sha(F/'attempt002/receipt.json'),'source_scope_review_sha256':sha(F/'source-scope-review.json'),'report_sha256':sha(F/'REPORT.md'),'core_actual_scope':'saved aggregate and source/count audit; independent exact204 polynomial targets; no unavailable per-case actual residual reconstruction','no_new_query_compile_matrix_eigen_evolution':True,'preserved_original_failure_and_independent_parse_failure':True}
(F/'index.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(F),'index_sha256':sha(F/'index.json'),'file_count':len(records),'bytes':obj['bytes'],'receipt_sha256':obj['receipt_sha256'],'source_scope_review_sha256':obj['source_scope_review_sha256'],'report_sha256':obj['report_sha256']},indent=2))
