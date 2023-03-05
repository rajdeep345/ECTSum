from utils import *

# Paraphrasing
# Doc - All lines that cover target summary sentences.
# Summ - Corresponding factually grounded sentences.
# Numbers in both documents and summaries are masked by placeholders.


def getMaskedLines(d_lines, s_lines):
	dlines_masked, slines_masked = [], []
	for dline, sline in zip(d_lines, s_lines):
		dlines_num = {}
		count = 1
		dline = getPPText(dline)
		for val in re.findall(pattern7, dline):
			if val not in dlines_num:
				dlines_num[val] = f'num-{num2words(count)}'
				count += 1

		vals = re.findall(pattern7, dline)
		for val in vals:
			if '.' in val and len([v for v in vals if val != v and val in v]) == 0:
				dline = dline.replace(val, dlines_num[val])
		for val in vals:
			if '.' in val:
				dline = dline.replace(val, dlines_num[val])		
		
		vals = re.findall(pattern7, dline)
		for val in vals:
			if len([v for v in vals if val != v and val in v]) == 0:
				dline = dline.replace(val, dlines_num[val])
		for val in vals:
			dline = dline.replace(val, dlines_num[val])
		dlines_masked.append(dline)
		
		sline = getPPText(sline)
		vals = re.findall(pattern7, sline)
		for val in vals:
			if '.' in val and len([v for v in vals if val != v and val in v]) == 0:
				sline = sline.replace(val, dlines_num[val]) if val in dlines_num else sline
		for val in vals:
			if '.' in val:
				sline = sline.replace(val, dlines_num[val]) if val in dlines_num else sline
		
		vals = re.findall(pattern7, sline)
		for val in vals:
			if len([v for v in vals if val != v and val in v]) == 0:
				sline = sline.replace(val, dlines_num[val]) if val in dlines_num else sline
		for val in vals:
			sline = sline.replace(val, dlines_num[val]) if val in dlines_num else sline
		slines_masked.append(sline)
	
	return dlines_masked, slines_masked


def prepare_data(dataPath, out_path, exp):
	source_path = f'{dataPath}/source/'
	target_path = f'{dataPath}/target/'
	if not os.path.isdir(f'{out_path}/source/'):
		os.makedirs(f'{out_path}/source/')
	if not os.path.isdir(f'{out_path}/target/'):
		os.makedirs(f'{out_path}/target/')
	for file in os.listdir(source_path):
		if file.endswith('.txt'):
			print(file)
			f_ect_in = open(f'{source_path}{file}', 'r')
			doc_lines = [line.strip() for line in f_ect_in.readlines()]
			f_summ_in = open(f'{summ_path}{file}', 'r')
			summ_lines = [line.strip() for line in f_summ_in.readlines()]
			assert len(doc_lines) == len(summ_lines)
			d_lines, s_lines = getMaskedLines(doc_lines, summ_lines)
			assert len(d_lines) == len(s_lines)
			doc_out = open(f'{out_path}/source/{file}', 'w')
			doc_out.write('\n'.join(d_lines))
			summ_out = open(f'{out_path}/target/{file}', 'w')
			summ_out.write('\n'.join(s_lines))
			doc_out.close()
			summ_out.close()

for split in ['train', 'val']:
	print(f'\n\n Preparing {split} data..\n')
	prepare_data(f'codes/ECT-BPS/ectbps_para/data/para/{split}', f'codes/ECT-BPS/ectbps_para/data/para_mask/{split}')
