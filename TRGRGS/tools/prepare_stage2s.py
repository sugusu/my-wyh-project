#!/usr/bin/env python3
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];scene=ROOT/'data/translab/scene_01';out=ROOT/'outputs/scene_01/stage2s';out.mkdir(parents=True,exist_ok=True)
old=[ROOT/'reports/stage2a_gt_free_consensus.json',ROOT/'reports/stage2a_gt_diagnostic.json',ROOT/'reports/stage2a_final_decision.json']+sorted((ROOT/'outputs/scene_01/stage2a_consensus').rglob('*'))
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in old if p.is_file()}
archive={'status':'CASE_PIXELWISE_CROSS_VIEW_SELECTION_FAIL','stage1_hypothesis_utility':True,'stage1_dispersion_reliability':True,'pixelwise_consensus_usable':False,'original_stage2b_authorized':False,'alternative_branch':'SPLIT_VIEW_COUNTERFACTUAL_RESPONSIBILITY','archived_sha256':hashes}
(ROOT/'reports/stage2a_archived_conclusion.json').write_text(json.dumps(archive,indent=2)+'\n')
all_names=sorted(p.name for p in (scene/'images').glob('*') if p.is_file());official_test=[n for i,n in enumerate(all_names) if i%8==0];original_train=[n for i,n in enumerate(all_names) if i%8!=0];assert len(all_names)==400 and len(original_train)==350
for tag,mod in [('a',0),('b',3)]:
 hold=[n for i,n in enumerate(original_train) if i%7==mod];train=[n for n in original_train if n not in set(hold)];assert not set(hold)&set(official_test)
 split_scene=ROOT/f'data/stage2s/split_{tag}/scene_01';split_scene.mkdir(parents=True,exist_ok=True)
 for child in ['images','delights','normals','normal','transparent_masks','transparent_mask','masks','meshes','transnormals']:
  dst=split_scene/child
  if not dst.exists():dst.symlink_to(scene/child)
 sparse=split_scene/'sparse';sparse.mkdir(exist_ok=True)
 for f in ['cameras.txt','cameras.bin','images.txt','images.bin','points3D.txt','points3D.bin']:
  dst=sparse/f
  if not dst.exists():dst.symlink_to(scene/'sparse'/f)
 # TSGS CamInfo.image_name is the filename stem, while the protocol manifest
 # deliberately records complete filenames.
 split={'train':[Path(n).stem for n in train],'test':[Path(n).stem for n in hold]};(split_scene/'split.json').write_text(json.dumps(split,indent=2)+'\n')
 canon=lambda x:'\n'.join(x)+'\n';manifest={'split':tag.upper(),'rule':f'holdout index % 7 == {mod}','source_camera_count':350,'train_count':len(train),'holdout_count':len(hold),'official_test_excluded_count':len(official_test),'train':train,'holdout':hold,'train_names_sha256':hashlib.sha256(canon(train).encode()).hexdigest(),'holdout_names_sha256':hashlib.sha256(canon(hold).encode()).hexdigest(),'split_json_sha256':hashlib.sha256((split_scene/'split.json').read_bytes()).hexdigest(),'source_scene':str(scene.resolve()),'split_scene':str(split_scene)}
 (out/f'split_{tag}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert not set(json.load(open(out/'split_a_manifest.json'))['holdout'])&set(json.load(open(out/'split_b_manifest.json'))['holdout'])
print(json.dumps({'split_a':{'train':300,'holdout':50},'split_b':{'train':300,'holdout':50},'overlap':0},indent=2))
