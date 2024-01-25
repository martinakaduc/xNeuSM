README for dataset KKI


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Node Label Conversion === 

Node labels were converted to integer values using this map:

Component 0:
	0	n_1
	1	n_187
	2	n_133
	3	n_137
	4	n_18
	5	n_46
	6	n_88
	7	n_91
	8	n_123
	9	n_188
	10	n_30
	11	n_65
	12	n_71
	13	n_73
	14	n_82
	15	n_90
	16	n_101
	17	n_113
	18	n_117
	19	n_183
	20	n_17
	21	n_139
	22	n_11
	23	n_12
	24	n_43
	25	n_83
	26	n_102
	27	n_57
	28	n_144
	29	n_130
	30	n_186
	31	n_178
	32	n_106
	33	n_16
	34	n_37
	35	n_41
	36	n_47
	37	n_94
	38	n_124
	39	n_131
	40	n_146
	41	n_153
	42	n_155
	43	n_168
	44	n_177
	45	n_161
	46	n_80
	47	n_8
	48	n_15
	49	n_158
	50	n_140
	51	n_114
	52	n_76
	53	n_121
	54	n_2
	55	n_179
	56	n_100
	57	n_180
	58	n_42
	59	n_169
	60	n_85
	61	n_132
	62	n_63
	63	n_7
	64	n_97
	65	n_145
	66	n_134
	67	n_152
	68	n_9
	69	n_27
	70	n_166
	71	n_111
	72	n_115
	73	n_108
	74	n_126
	75	n_147
	76	n_175
	77	n_6
	78	n_62
	79	n_141
	80	n_173
	81	n_154
	82	n_135
	83	n_55
	84	n_96
	85	n_99
	86	n_148
	87	n_156
	88	n_44
	89	n_72
	90	n_93
	91	n_87
	92	n_95
	93	n_185
	94	n_51
	95	n_24
	96	n_110
	97	n_23
	98	n_49
	99	n_136
	100	n_36
	101	n_170
	102	n_31
	103	n_25
	104	n_69
	105	n_3
	106	n_84
	107	n_142
	108	n_174
	109	n_10
	110	n_109
	111	n_4
	112	n_60
	113	n_45
	114	n_105
	115	n_119
	116	n_162
	117	n_127
	118	n_38
	119	n_74
	120	n_59
	121	n_70
	122	n_122
	123	n_39
	124	n_53
	125	n_5
	126	n_128
	127	n_13
	128	n_184
	129	n_68
	130	n_66
	131	n_129
	132	n_160
	133	n_29
	134	n_92
	135	n_103
	136	n_176
	137	n_138
	138	n_19
	139	n_52
	140	n_181
	141	n_182
	142	n_157
	143	n_164
	144	n_120
	145	n_149
	146	n_33
	147	n_118
	148	n_172
	149	n_143
	150	n_50
	151	n_151
	152	n_56
	153	n_58
	154	n_86
	155	n_79
	156	n_75
	157	n_67
	158	n_78
	159	n_81
	160	n_54
	161	n_150
	162	n_77
	163	n_35
	164	n_167
	165	n_171
	166	n_189
	167	n_107
	168	n_0
	169	n_159
	170	n_89
	171	n_32
	172	n_26
	173	n_14
	174	n_21
	175	n_98
	176	n_125
	177	n_116
	178	n_28
	179	n_48
	180	n_104
	181	n_20
	182	n_64
	183	n_40
	184	n_163
	185	n_22
	186	n_34
	187	n_165
	188	n_61
	189	n_112


=== References ===
https://github.com/shiruipan/graph_datasets/tree/master/Graph_Repository

=== Previous Use of the Dataset ===
Shirui Pan, Jia Wu, Xingquan Zhu, Guodong Long, and Chengqi Zhang. " Task Sensitive Feature Exploration and Learning for Multi-Task Graph Classification."  IEEE Trans. Cybernetics (TCYB) 47(3): 744-758 (2017)

