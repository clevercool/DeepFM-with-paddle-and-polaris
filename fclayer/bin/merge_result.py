f_index=open('set.b.txt','r')
f_res=open('setb_offset_2.txt','r')
f_out=open('predic.txt','w')
res_list=list()
index_list=list()
for line in f_res:
	line = float(line)
	line=round(line,6)
	res_list.append(line)

for line in f_index:
	line=line.split(',')
	index_list.append(line[0])
	
for i in range(0,len(index_list)):
	f_out.write(str(index_list[i])+','+str(res_list[i])+str('\n'))

	
