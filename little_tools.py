import os
from inspect import isfunction

def aa():
	return 0

class adfas():
	def __init__(self):
		pass
	def aaa(self):
		return 0
AS = adfas()

def print_obj(who, name, depth=0, max_bro=False, max_len=False):
	if type(who) == list or type(who) == tuple:
		if max_bro and len(who)>max_bro:
			print_obj(who[0:max_bro//2-1]+['...']+who[len(who)-max_bro//2+1:], name, depth, max_bro, max_len)
		else:
			print("  "*depth, "L",name)
			for a in who:
				print_obj(a, "", depth+1, max_bro, max_len)
	elif type(who) == dict:
		print("  "*depth, "L",name)
		for n, a in who.items():
			print_obj(a, n, depth+1, max_bro, max_len)
	elif not hasattr(who, '__dict__'):
		print("  "*depth, "L", name, who)
	else:
		for n, a in who.__dict__.items():
			if n[0]!='_':
				print_obj(a, n, depth+1, max_bro, max_len)
