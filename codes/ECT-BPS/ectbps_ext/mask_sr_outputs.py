from ect_utils import *

# SummaRuNNer
# Mask the numerical values in the extractive summaries returned by SummaRuNNer.


def getMaskedLines(d_lines):
	dlines_masked = []
	for dline in d_lines:
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
	
	return dlines_masked


def mask_outputs(predPath, out_path):
	if not os.path.isdir(f'{out_path}'):
		os.makedirs(f'{out_path}')
	for file in os.listdir(predPath):
		if file.endswith('.txt'):
			f_pred_in = open(f'{predPath}/{file}', 'r')
			lines = [line.strip() for line in f_pred_in.readlines()]
			masked_lines = getMaskedLines(lines)
			with open(f'{out_path}/{file}', 'w') as f_out:
				f_out.write('\n'.join(masked_lines))


mask_outputs('outputs/hyp', 'outputs/hyp_mask')
