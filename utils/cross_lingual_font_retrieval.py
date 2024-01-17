import os
from utils.initialize_font_data import retrieve_font_path, font_dir, cj_font_dir

exclusive_cjk_font_paths = [
  'M+A1_heavy-50-1.2.otf',
  'DroidSansFallback.ttf',
  'M+A1_heavy-10-1.2.otf',
  'M+A1_light-24-2.0.otf',
  'M+A1_heavy-42-1.2.otf',
  'M+A1_heavy-58-1.2.otf',
  'M+A1_regular-35-2.0.otf',
  'NewFont-Regular.otf',
  'M+A1_heavy-35-2.0.otf',
  'YuNaFont_P.ttf',
  'Pomeranian-Regular.ttf',
  'KodomoRounded.otf',
  'g_squarebold_free_010.ttf',
  'GenEiAntiquePv5-M.ttf',
  'HanyiSentySpringBrush.ttf',
  'FZJianZhi-t.TTF',
  'SentyGoldenBell.ttf',
  'jetlink-boldpeakpop.TTF',
  'FZTeCuGuangHui.TTF',
  'ftiebihei.TTF',
  'HD-HPMST.ttf',
  'YuNaFont.ttf',
  'hamburger-font.ttf',
  'mini-jianzhi.TTF',
  'DFGirlW7-B5.TTC'
]
roman_font_dir_path = font_dir
cjk_font_dir_path = cj_font_dir
all_font_paths = [os.path.join(cjk_font_dir_path, font_name) for font_name in os.listdir(cjk_font_dir_path)]
inclusive_cjk_font_paths = [font_path for font_path in all_font_paths if os.path.basename(font_path) not in exclusive_cjk_font_paths]
roman_ref_font_names_path = '../attributeData/cross-lingual-font-info/ref_roman.txt'
cjk_ref_font_names_path = '../attributeData/cross-lingual-font-info/ref_cjk.txt'
with open(roman_ref_font_names_path, 'r') as f:
  roman_ref_font_names = f.read().splitlines()
with open(cjk_ref_font_names_path, 'r') as f:
  cjk_ref_font_names = f.read().splitlines()
roman_ref_font_paths = [retrieve_font_path(font_name, font_dir=roman_font_dir_path) for font_name in roman_ref_font_names]
cjk_ref_font_paths = [retrieve_font_path(font_name, font_dir=cjk_font_dir_path) for font_name in cjk_ref_font_names]