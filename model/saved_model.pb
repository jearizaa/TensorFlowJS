??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
o
identifiersVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameidentifiers
h
identifiers/Read/ReadVariableOpReadVariableOpidentifiers*
_output_shapes	
:?*
dtype0
q

candidatesVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *
shared_name
candidates
j
candidates/Read/ReadVariableOpReadVariableOp
candidates*
_output_shapes
:	? *
dtype0
?
embedding_11/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *(
shared_nameembedding_11/embeddings
?
+embedding_11/embeddings/Read/ReadVariableOpReadVariableOpembedding_11/embeddings*
_output_shapes
:	? *
dtype0
?
string_lookup_13_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_9070*
value_dtype0	
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_17956

NoOpNoOp^PartitionedCall
?
Kstring_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_13_index_table*
Tkeys0*
Tvalues0	*/
_class%
#!loc:@string_lookup_13_index_table*
_output_shapes

::
?
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
query_model
identifiers
_identifiers

candidates
_candidates
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
	variables
trainable_variables
regularization_losses
	keras_api
GE
VARIABLE_VALUEidentifiers&identifiers/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE
candidates%candidates/.ATTRIBUTES/VARIABLE_VALUE

1
2
3

0
 
?
layer_regularization_losses

layers
	variables
layer_metrics
metrics
trainable_variables
non_trainable_variables
regularization_losses
 
0
state_variables

_table
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api

1

0
 
?
layer_regularization_losses

layers
	variables
layer_metrics
metrics
trainable_variables
 non_trainable_variables
regularization_losses
SQ
VARIABLE_VALUEembedding_11/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
 

0
 
 

1
2
 
DB
table9query_model/layer_with_weights-0/_table/.ATTRIBUTES/table
 

0

0
 
?
!layer_regularization_losses

"layers
	variables
#layer_metrics
$metrics
trainable_variables
%non_trainable_variables
regularization_losses
 

	0

1
 
 
 
 
 
 
 
 
r
serving_default_input_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1string_lookup_13_index_tableConstembedding_11/embeddings
candidatesidentifiers*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_17664
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameidentifiers/Read/ReadVariableOpcandidates/Read/ReadVariableOp+embedding_11/embeddings/Read/ReadVariableOpKstring_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2Mstring_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2:1Const_1*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_17996
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameidentifiers
candidatesembedding_11/embeddingsstring_lookup_13_index_table*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_18018??
?
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17871

inputsb
^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handlec
_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	9
5embedding_11_embedding_lookup_readvariableop_resource
identity??,embedding_11/embedding_lookup/ReadVariableOp?Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleinputs_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2S
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
,embedding_11/embedding_lookup/ReadVariableOpReadVariableOp5embedding_11_embedding_lookup_readvariableop_resource*
_output_shapes
:	? *
dtype02.
,embedding_11/embedding_lookup/ReadVariableOp?
"embedding_11/embedding_lookup/axisConst*?
_class5
31loc:@embedding_11/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2$
"embedding_11/embedding_lookup/axis?
embedding_11/embedding_lookupGatherV24embedding_11/embedding_lookup/ReadVariableOp:value:0Zstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:0+embedding_11/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*?
_class5
31loc:@embedding_11/embedding_lookup/ReadVariableOp*'
_output_shapes
:????????? 2
embedding_11/embedding_lookup?
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0*
T0*'
_output_shapes
:????????? 2(
&embedding_11/embedding_lookup/Identity?
IdentityIdentity/embedding_11/embedding_lookup/Identity:output:0-^embedding_11/embedding_lookup/ReadVariableOpR^string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*,
_input_shapes
:?????????:: :2\
,embedding_11/embedding_lookup/ReadVariableOp,embedding_11/embedding_lookup/ReadVariableOp2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?)
?
 __inference__wrapped_model_17442
input_1~
zbrute_force_1_sequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handle
{brute_force_1_sequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	U
Qbrute_force_1_sequential_11_embedding_11_embedding_lookup_readvariableop_resource0
,brute_force_1_matmul_readvariableop_resource!
brute_force_1_gather_resource
identity

identity_1??brute_force_1/Gather?#brute_force_1/MatMul/ReadVariableOp?Hbrute_force_1/sequential_11/embedding_11/embedding_lookup/ReadVariableOp?mbrute_force_1/sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
mbrute_force_1/sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2zbrute_force_1_sequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleinput_1{brute_force_1_sequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2o
mbrute_force_1/sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
Hbrute_force_1/sequential_11/embedding_11/embedding_lookup/ReadVariableOpReadVariableOpQbrute_force_1_sequential_11_embedding_11_embedding_lookup_readvariableop_resource*
_output_shapes
:	? *
dtype02J
Hbrute_force_1/sequential_11/embedding_11/embedding_lookup/ReadVariableOp?
>brute_force_1/sequential_11/embedding_11/embedding_lookup/axisConst*[
_classQ
OMloc:@brute_force_1/sequential_11/embedding_11/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2@
>brute_force_1/sequential_11/embedding_11/embedding_lookup/axis?
9brute_force_1/sequential_11/embedding_11/embedding_lookupGatherV2Pbrute_force_1/sequential_11/embedding_11/embedding_lookup/ReadVariableOp:value:0vbrute_force_1/sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:0Gbrute_force_1/sequential_11/embedding_11/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*[
_classQ
OMloc:@brute_force_1/sequential_11/embedding_11/embedding_lookup/ReadVariableOp*'
_output_shapes
:????????? 2;
9brute_force_1/sequential_11/embedding_11/embedding_lookup?
Bbrute_force_1/sequential_11/embedding_11/embedding_lookup/IdentityIdentityBbrute_force_1/sequential_11/embedding_11/embedding_lookup:output:0*
T0*'
_output_shapes
:????????? 2D
Bbrute_force_1/sequential_11/embedding_11/embedding_lookup/Identity?
#brute_force_1/MatMul/ReadVariableOpReadVariableOp,brute_force_1_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02%
#brute_force_1/MatMul/ReadVariableOp?
brute_force_1/MatMulMatMulKbrute_force_1/sequential_11/embedding_11/embedding_lookup/Identity:output:0+brute_force_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(2
brute_force_1/MatMulr
brute_force_1/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2
brute_force_1/TopKV2/k?
brute_force_1/TopKV2TopKV2brute_force_1/MatMul:product:0brute_force_1/TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
brute_force_1/TopKV2?
brute_force_1/GatherResourceGatherbrute_force_1_gather_resourcebrute_force_1/TopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
brute_force_1/Gather?
brute_force_1/IdentityIdentitybrute_force_1/Gather:output:0*
T0*'
_output_shapes
:?????????
2
brute_force_1/Identity?
IdentityIdentitybrute_force_1/TopKV2:values:0^brute_force_1/Gather$^brute_force_1/MatMul/ReadVariableOpI^brute_force_1/sequential_11/embedding_11/embedding_lookup/ReadVariableOpn^brute_force_1/sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identitybrute_force_1/Identity:output:0^brute_force_1/Gather$^brute_force_1/MatMul/ReadVariableOpI^brute_force_1/sequential_11/embedding_11/embedding_lookup/ReadVariableOpn^brute_force_1/sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!:?????????:: :::2,
brute_force_1/Gatherbrute_force_1/Gather2J
#brute_force_1/MatMul/ReadVariableOp#brute_force_1/MatMul/ReadVariableOp2?
Hbrute_force_1/sequential_11/embedding_11/embedding_lookup/ReadVariableOpHbrute_force_1/sequential_11/embedding_11/embedding_lookup/ReadVariableOp2?
mbrute_force_1/sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2mbrute_force_1/sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?	
?
__inference_restore_fn_17951
restored_tensors_0
restored_tensors_1	O
Kstring_lookup_13_index_table_table_restore_lookuptableimportv2_table_handle
identity??>string_lookup_13_index_table_table_restore/LookupTableImportV2?
>string_lookup_13_index_table_table_restore/LookupTableImportV2LookupTableImportV2Kstring_lookup_13_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2@
>string_lookup_13_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0?^string_lookup_13_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:::2?
>string_lookup_13_index_table_table_restore/LookupTableImportV2>string_lookup_13_index_table_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?	
?
-__inference_brute_force_1_layer_call_fn_17847
queries
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_brute_force_1_layer_call_and_return_conditional_losses_176132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!:?????????:: :::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?	
?
-__inference_brute_force_1_layer_call_fn_17830
queries
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_brute_force_1_layer_call_and_return_conditional_losses_176132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!:?????????:: :::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17859

inputsb
^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handlec
_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	9
5embedding_11_embedding_lookup_readvariableop_resource
identity??,embedding_11/embedding_lookup/ReadVariableOp?Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleinputs_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2S
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
,embedding_11/embedding_lookup/ReadVariableOpReadVariableOp5embedding_11_embedding_lookup_readvariableop_resource*
_output_shapes
:	? *
dtype02.
,embedding_11/embedding_lookup/ReadVariableOp?
"embedding_11/embedding_lookup/axisConst*?
_class5
31loc:@embedding_11/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2$
"embedding_11/embedding_lookup/axis?
embedding_11/embedding_lookupGatherV24embedding_11/embedding_lookup/ReadVariableOp:value:0Zstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:0+embedding_11/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*?
_class5
31loc:@embedding_11/embedding_lookup/ReadVariableOp*'
_output_shapes
:????????? 2
embedding_11/embedding_lookup?
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0*
T0*'
_output_shapes
:????????? 2(
&embedding_11/embedding_lookup/Identity?
IdentityIdentity/embedding_11/embedding_lookup/Identity:output:0-^embedding_11/embedding_lookup/ReadVariableOpR^string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*,
_input_shapes
:?????????:: :2\
,embedding_11/embedding_lookup/ReadVariableOp,embedding_11/embedding_lookup/ReadVariableOp2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
.
__inference__initializer_17919
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
r
,__inference_embedding_11_layer_call_fn_17909

inputs	
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_11_layer_call_and_return_conditional_losses_174582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_11_layer_call_fn_17882

inputs
unknown
	unknown_0	
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_11_layer_call_and_return_conditional_losses_174942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*,
_input_shapes
:?????????:: :22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17613
queries
sequential_11_17595
sequential_11_17597	
sequential_11_17599"
matmul_readvariableop_resource
gather_resource

identity_1

identity_2??Gather?MatMul/ReadVariableOp?%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_11_17595sequential_11_17597sequential_11_17599*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_11_layer_call_and_return_conditional_losses_175152'
%sequential_11/StatefulPartitionedCall?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOp?
MatMulMatMul.sequential_11/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1IdentityTopKV2:values:0^Gather^MatMul/ReadVariableOp&^sequential_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2IdentityIdentity:output:0^Gather^MatMul/ReadVariableOp&^sequential_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*4
_input_shapes#
!:?????????:: :::2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17471
string_lookup_13_inputb
^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handlec
_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	
embedding_11_17467
identity??$embedding_11/StatefulPartitionedCall?Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handlestring_lookup_13_input_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2S
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
$embedding_11/StatefulPartitionedCallStatefulPartitionedCallZstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:0embedding_11_17467*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_11_layer_call_and_return_conditional_losses_174582&
$embedding_11/StatefulPartitionedCall?
IdentityIdentity-embedding_11/StatefulPartitionedCall:output:0%^embedding_11/StatefulPartitionedCallR^string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*,
_input_shapes
:?????????:: :2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:[ W
#
_output_shapes
:?????????
0
_user_specified_namestring_lookup_13_input:

_output_shapes
: 
?
?
__inference_save_fn_17943
checkpoint_key\
Xstring_lookup_13_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Kstring_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2?
Kstring_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Xstring_lookup_13_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2M
Kstring_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0L^string_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0L^string_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityRstring_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2:keys:0L^string_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_2?

Identity_3Identity	add_1:z:0L^string_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0L^string_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityTstring_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2:values:0L^string_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Kstring_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2Kstring_lookup_13_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
*
__inference_<lambda>_17956
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17515

inputsb
^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handlec
_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	
embedding_11_17511
identity??$embedding_11/StatefulPartitionedCall?Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleinputs_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2S
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
$embedding_11/StatefulPartitionedCallStatefulPartitionedCallZstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:0embedding_11_17511*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_11_layer_call_and_return_conditional_losses_174582&
$embedding_11/StatefulPartitionedCall?
IdentityIdentity-embedding_11/StatefulPartitionedCall:output:0%^embedding_11/StatefulPartitionedCallR^string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*,
_input_shapes
:?????????:: :2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
-__inference_sequential_11_layer_call_fn_17893

inputs
unknown
	unknown_0	
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_11_layer_call_and_return_conditional_losses_175152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*,
_input_shapes
:?????????:: :22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
-__inference_brute_force_1_layer_call_fn_17769
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_brute_force_1_layer_call_and_return_conditional_losses_176132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!:?????????:: :::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?#
?
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17713
input_1p
lsequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleq
msequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	G
Csequential_11_embedding_11_embedding_lookup_readvariableop_resource"
matmul_readvariableop_resource
gather_resource

identity_1

identity_2??Gather?MatMul/ReadVariableOp?:sequential_11/embedding_11/embedding_lookup/ReadVariableOp?_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2lsequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleinput_1msequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2a
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
:sequential_11/embedding_11/embedding_lookup/ReadVariableOpReadVariableOpCsequential_11_embedding_11_embedding_lookup_readvariableop_resource*
_output_shapes
:	? *
dtype02<
:sequential_11/embedding_11/embedding_lookup/ReadVariableOp?
0sequential_11/embedding_11/embedding_lookup/axisConst*M
_classC
A?loc:@sequential_11/embedding_11/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/embedding_11/embedding_lookup/axis?
+sequential_11/embedding_11/embedding_lookupGatherV2Bsequential_11/embedding_11/embedding_lookup/ReadVariableOp:value:0hsequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:09sequential_11/embedding_11/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*M
_classC
A?loc:@sequential_11/embedding_11/embedding_lookup/ReadVariableOp*'
_output_shapes
:????????? 2-
+sequential_11/embedding_11/embedding_lookup?
4sequential_11/embedding_11/embedding_lookup/IdentityIdentity4sequential_11/embedding_11/embedding_lookup:output:0*
T0*'
_output_shapes
:????????? 26
4sequential_11/embedding_11/embedding_lookup/Identity?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOp?
MatMulMatMul=sequential_11/embedding_11/embedding_lookup/Identity:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1IdentityTopKV2:values:0^Gather^MatMul/ReadVariableOp;^sequential_11/embedding_11/embedding_lookup/ReadVariableOp`^sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2IdentityIdentity:output:0^Gather^MatMul/ReadVariableOp;^sequential_11/embedding_11/embedding_lookup/ReadVariableOp`^sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????
2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*4
_input_shapes#
!:?????????:: :::2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2x
:sequential_11/embedding_11/embedding_lookup/ReadVariableOp:sequential_11/embedding_11/embedding_lookup/ReadVariableOp2?
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?#
?
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17735
input_1p
lsequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleq
msequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	G
Csequential_11_embedding_11_embedding_lookup_readvariableop_resource"
matmul_readvariableop_resource
gather_resource

identity_1

identity_2??Gather?MatMul/ReadVariableOp?:sequential_11/embedding_11/embedding_lookup/ReadVariableOp?_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2lsequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleinput_1msequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2a
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
:sequential_11/embedding_11/embedding_lookup/ReadVariableOpReadVariableOpCsequential_11_embedding_11_embedding_lookup_readvariableop_resource*
_output_shapes
:	? *
dtype02<
:sequential_11/embedding_11/embedding_lookup/ReadVariableOp?
0sequential_11/embedding_11/embedding_lookup/axisConst*M
_classC
A?loc:@sequential_11/embedding_11/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/embedding_11/embedding_lookup/axis?
+sequential_11/embedding_11/embedding_lookupGatherV2Bsequential_11/embedding_11/embedding_lookup/ReadVariableOp:value:0hsequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:09sequential_11/embedding_11/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*M
_classC
A?loc:@sequential_11/embedding_11/embedding_lookup/ReadVariableOp*'
_output_shapes
:????????? 2-
+sequential_11/embedding_11/embedding_lookup?
4sequential_11/embedding_11/embedding_lookup/IdentityIdentity4sequential_11/embedding_11/embedding_lookup:output:0*
T0*'
_output_shapes
:????????? 26
4sequential_11/embedding_11/embedding_lookup/Identity?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOp?
MatMulMatMul=sequential_11/embedding_11/embedding_lookup/Identity:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1IdentityTopKV2:values:0^Gather^MatMul/ReadVariableOp;^sequential_11/embedding_11/embedding_lookup/ReadVariableOp`^sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2IdentityIdentity:output:0^Gather^MatMul/ReadVariableOp;^sequential_11/embedding_11/embedding_lookup/ReadVariableOp`^sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????
2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*4
_input_shapes#
!:?????????:: :::2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2x
:sequential_11/embedding_11/embedding_lookup/ReadVariableOp:sequential_11/embedding_11/embedding_lookup/ReadVariableOp2?
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?

?
G__inference_embedding_11_layer_call_and_return_conditional_losses_17902

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	? *
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:????????? 2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:????????? 2
embedding_lookup/Identity?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17494

inputsb
^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handlec
_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	
embedding_11_17490
identity??$embedding_11/StatefulPartitionedCall?Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleinputs_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2S
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
$embedding_11/StatefulPartitionedCallStatefulPartitionedCallZstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:0embedding_11_17490*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_11_layer_call_and_return_conditional_losses_174582&
$embedding_11/StatefulPartitionedCall?
IdentityIdentity-embedding_11/StatefulPartitionedCall:output:0%^embedding_11/StatefulPartitionedCallR^string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*,
_input_shapes
:?????????:: :2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
,
__inference__destroyer_17924
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?#
?
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17813
queriesp
lsequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleq
msequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	G
Csequential_11_embedding_11_embedding_lookup_readvariableop_resource"
matmul_readvariableop_resource
gather_resource

identity_1

identity_2??Gather?MatMul/ReadVariableOp?:sequential_11/embedding_11/embedding_lookup/ReadVariableOp?_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2lsequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handlequeriesmsequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2a
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
:sequential_11/embedding_11/embedding_lookup/ReadVariableOpReadVariableOpCsequential_11_embedding_11_embedding_lookup_readvariableop_resource*
_output_shapes
:	? *
dtype02<
:sequential_11/embedding_11/embedding_lookup/ReadVariableOp?
0sequential_11/embedding_11/embedding_lookup/axisConst*M
_classC
A?loc:@sequential_11/embedding_11/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/embedding_11/embedding_lookup/axis?
+sequential_11/embedding_11/embedding_lookupGatherV2Bsequential_11/embedding_11/embedding_lookup/ReadVariableOp:value:0hsequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:09sequential_11/embedding_11/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*M
_classC
A?loc:@sequential_11/embedding_11/embedding_lookup/ReadVariableOp*'
_output_shapes
:????????? 2-
+sequential_11/embedding_11/embedding_lookup?
4sequential_11/embedding_11/embedding_lookup/IdentityIdentity4sequential_11/embedding_11/embedding_lookup:output:0*
T0*'
_output_shapes
:????????? 26
4sequential_11/embedding_11/embedding_lookup/Identity?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOp?
MatMulMatMul=sequential_11/embedding_11/embedding_lookup/Identity:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1IdentityTopKV2:values:0^Gather^MatMul/ReadVariableOp;^sequential_11/embedding_11/embedding_lookup/ReadVariableOp`^sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2IdentityIdentity:output:0^Gather^MatMul/ReadVariableOp;^sequential_11/embedding_11/embedding_lookup/ReadVariableOp`^sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????
2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*4
_input_shapes#
!:?????????:: :::2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2x
:sequential_11/embedding_11/embedding_lookup/ReadVariableOp:sequential_11/embedding_11/embedding_lookup/ReadVariableOp2?
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
-__inference_sequential_11_layer_call_fn_17503
string_lookup_13_input
unknown
	unknown_0	
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_13_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_11_layer_call_and_return_conditional_losses_174942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*,
_input_shapes
:?????????:: :22
StatefulPartitionedCallStatefulPartitionedCall:[ W
#
_output_shapes
:?????????
0
_user_specified_namestring_lookup_13_input:

_output_shapes
: 
?
L
__inference__creator_17914
identity??string_lookup_13_index_table?
string_lookup_13_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_9070*
value_dtype0	2
string_lookup_13_index_table?
IdentityIdentity+string_lookup_13_index_table:table_handle:0^string_lookup_13_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2<
string_lookup_13_index_tablestring_lookup_13_index_table
?
?
!__inference__traced_restore_18018
file_prefix 
assignvariableop_identifiers!
assignvariableop_1_candidates.
*assignvariableop_2_embedding_11_embeddings_
[string_lookup_13_index_table_table_restore_lookuptableimportv2_string_lookup_13_index_table

identity_4??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?>string_lookup_13_index_table_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB>query_model/layer_with_weights-0/_table/.ATTRIBUTES/table-keysB@query_model/layer_with_weights-0/_table/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_identifiersIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_candidatesIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_embedding_11_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2?
>string_lookup_13_index_table_table_restore/LookupTableImportV2LookupTableImportV2[string_lookup_13_index_table_table_restore_lookuptableimportv2_string_lookup_13_index_tableRestoreV2:tensors:3RestoreV2:tensors:4*	
Tin0*

Tout0	*/
_class%
#!loc:@string_lookup_13_index_table*
_output_shapes
 2@
>string_lookup_13_index_table_table_restore/LookupTableImportV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp?^string_lookup_13_index_table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_3?

Identity_4IdentityIdentity_3:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2?^string_lookup_13_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity_4"!

identity_4Identity_4:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22?
>string_lookup_13_index_table_table_restore/LookupTableImportV2>string_lookup_13_index_table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:51
/
_class%
#!loc:@string_lookup_13_index_table
?
?
__inference__traced_save_17996
file_prefix*
&savev2_identifiers_read_readvariableop)
%savev2_candidates_read_readvariableop6
2savev2_embedding_11_embeddings_read_readvariableopV
Rsavev2_string_lookup_13_index_table_lookup_table_export_values_lookuptableexportv2X
Tsavev2_string_lookup_13_index_table_lookup_table_export_values_lookuptableexportv2_1	
savev2_const_1

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB>query_model/layer_with_weights-0/_table/.ATTRIBUTES/table-keysB@query_model/layer_with_weights-0/_table/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_identifiers_read_readvariableop%savev2_candidates_read_readvariableop2savev2_embedding_11_embeddings_read_readvariableopRsavev2_string_lookup_13_index_table_lookup_table_export_values_lookuptableexportv2Tsavev2_string_lookup_13_index_table_lookup_table_export_values_lookuptableexportv2_1savev2_const_1"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
): :?:	? :	? ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:?:%!

_output_shapes
:	? :%!

_output_shapes
:	? :

_output_shapes
::

_output_shapes
::

_output_shapes
: 
?
?
-__inference_sequential_11_layer_call_fn_17524
string_lookup_13_input
unknown
	unknown_0	
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_13_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_11_layer_call_and_return_conditional_losses_175152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*,
_input_shapes
:?????????:: :22
StatefulPartitionedCallStatefulPartitionedCall:[ W
#
_output_shapes
:?????????
0
_user_specified_namestring_lookup_13_input:

_output_shapes
: 
?	
?
#__inference_signature_wrapper_17664
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_174422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!:?????????:: :::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?	
?
-__inference_brute_force_1_layer_call_fn_17752
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_brute_force_1_layer_call_and_return_conditional_losses_176132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!:?????????:: :::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?

?
G__inference_embedding_11_layer_call_and_return_conditional_losses_17458

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	? *
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:????????? 2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:????????? 2
embedding_lookup/Identity?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?#
?
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17791
queriesp
lsequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handleq
msequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	G
Csequential_11_embedding_11_embedding_lookup_readvariableop_resource"
matmul_readvariableop_resource
gather_resource

identity_1

identity_2??Gather?MatMul/ReadVariableOp?:sequential_11/embedding_11/embedding_lookup/ReadVariableOp?_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2lsequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handlequeriesmsequential_11_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2a
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
:sequential_11/embedding_11/embedding_lookup/ReadVariableOpReadVariableOpCsequential_11_embedding_11_embedding_lookup_readvariableop_resource*
_output_shapes
:	? *
dtype02<
:sequential_11/embedding_11/embedding_lookup/ReadVariableOp?
0sequential_11/embedding_11/embedding_lookup/axisConst*M
_classC
A?loc:@sequential_11/embedding_11/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/embedding_11/embedding_lookup/axis?
+sequential_11/embedding_11/embedding_lookupGatherV2Bsequential_11/embedding_11/embedding_lookup/ReadVariableOp:value:0hsequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:09sequential_11/embedding_11/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*M
_classC
A?loc:@sequential_11/embedding_11/embedding_lookup/ReadVariableOp*'
_output_shapes
:????????? 2-
+sequential_11/embedding_11/embedding_lookup?
4sequential_11/embedding_11/embedding_lookup/IdentityIdentity4sequential_11/embedding_11/embedding_lookup:output:0*
T0*'
_output_shapes
:????????? 26
4sequential_11/embedding_11/embedding_lookup/Identity?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOp?
MatMulMatMul=sequential_11/embedding_11/embedding_lookup/Identity:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(2
MatMulV
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
2

TopKV2/k?
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
2
TopKV2?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype02
Gatherc
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1IdentityTopKV2:values:0^Gather^MatMul/ReadVariableOp;^sequential_11/embedding_11/embedding_lookup/ReadVariableOp`^sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2IdentityIdentity:output:0^Gather^MatMul/ReadVariableOp;^sequential_11/embedding_11/embedding_lookup/ReadVariableOp`^sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????
2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*4
_input_shapes#
!:?????????:: :::2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2x
:sequential_11/embedding_11/embedding_lookup/ReadVariableOp:sequential_11/embedding_11/embedding_lookup/ReadVariableOp2?
_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2_sequential_11/string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17481
string_lookup_13_inputb
^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handlec
_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value	
embedding_11_17477
identity??$embedding_11/StatefulPartitionedCall?Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2LookupTableFindV2^string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_table_handlestring_lookup_13_input_string_lookup_13_string_lookup_13_index_table_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2S
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2?
$embedding_11/StatefulPartitionedCallStatefulPartitionedCallZstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:values:0embedding_11_17477*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_11_layer_call_and_return_conditional_losses_174582&
$embedding_11/StatefulPartitionedCall?
IdentityIdentity-embedding_11/StatefulPartitionedCall:output:0%^embedding_11/StatefulPartitionedCallR^string_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*,
_input_shapes
:?????????:: :2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall2?
Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2Qstring_lookup_13/string_lookup_13_index_table_lookup_table_find/LookupTableFindV2:[ W
#
_output_shapes
:?????????
0
_user_specified_namestring_lookup_13_input:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input_1,
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????
<
output_20
StatefulPartitionedCall:1?????????
tensorflow/serving/predict:?s
?
query_model
identifiers
_identifiers

candidates
_candidates
	variables
trainable_variables
regularization_losses
	keras_api

signatures
*&&call_and_return_all_conditional_losses
'__call__
(_default_save_signature
)query_with_exclusions"?
_tf_keras_model?{"class_name": "BruteForce", "name": "brute_force_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "BruteForce"}}
?
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
	variables
trainable_variables
regularization_losses
	keras_api
*,&call_and_return_all_conditional_losses
-__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "string", "sparse": false, "ragged": false, "name": "string_lookup_13_input"}}, {"class_name": "StringLookup", "config": {"name": "string_lookup_13", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "encoding": "utf-8"}}, {"class_name": "Embedding", "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 944, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "string", "sparse": false, "ragged": false, "name": "string_lookup_13_input"}}, {"class_name": "StringLookup", "config": {"name": "string_lookup_13", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "encoding": "utf-8"}}, {"class_name": "Embedding", "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 944, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}]}}}
:?2identifiers
:	? 2
candidates
5
1
2
3"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_regularization_losses

layers
	variables
layer_metrics
metrics
trainable_variables
non_trainable_variables
regularization_losses
'__call__
(_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
,
.serving_default"
signature_map
?
state_variables

_table
	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup_13", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_13", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "encoding": "utf-8"}}
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
*/&call_and_return_all_conditional_losses
0__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 944, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
'
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_regularization_losses

layers
	variables
layer_metrics
metrics
trainable_variables
 non_trainable_variables
regularization_losses
-__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
*:(	? 2embedding_11/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
O
1_create_resource
2_initialize
3_destroy_resourceR Z
table*+
"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
!layer_regularization_losses

"layers
	variables
#layer_metrics
$metrics
trainable_variables
%non_trainable_variables
regularization_losses
0__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17713
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17813
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17735
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17791?
???
FullArgSpec/
args'?$
jself
	jqueries
jk

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_brute_force_1_layer_call_fn_17830
-__inference_brute_force_1_layer_call_fn_17769
-__inference_brute_force_1_layer_call_fn_17752
-__inference_brute_force_1_layer_call_fn_17847?
???
FullArgSpec/
args'?$
jself
	jqueries
jk

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
 __inference__wrapped_model_17442?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *"?
?
input_1?????????
?2??
???
FullArgSpec1
args)?&
jself
	jqueries
j
exclusions
jk
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_save_fn_17943checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_17951restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
?2?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17481
H__inference_sequential_11_layer_call_and_return_conditional_losses_17859
H__inference_sequential_11_layer_call_and_return_conditional_losses_17471
H__inference_sequential_11_layer_call_and_return_conditional_losses_17871?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_11_layer_call_fn_17882
-__inference_sequential_11_layer_call_fn_17503
-__inference_sequential_11_layer_call_fn_17524
-__inference_sequential_11_layer_call_fn_17893?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_17664input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_11_layer_call_and_return_conditional_losses_17902?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_11_layer_call_fn_17909?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_17914?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_17919?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_17924?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
	J
Const6
__inference__creator_17914?

? 
? "? 8
__inference__destroyer_17924?

? 
? "? :
__inference__initializer_17919?

? 
? "? ?
 __inference__wrapped_model_17442?4,?)
"?
?
input_1?????????
? "c?`
.
output_1"?
output_1?????????

.
output_2"?
output_2?????????
?
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17713?44?1
*?'
?
input_1?????????

 
p
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17735?44?1
*?'
?
input_1?????????

 
p 
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17791?44?1
*?'
?
queries?????????

 
p
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
H__inference_brute_force_1_layer_call_and_return_conditional_losses_17813?44?1
*?'
?
queries?????????

 
p 
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
-__inference_brute_force_1_layer_call_fn_17752|44?1
*?'
?
input_1?????????

 
p
? "=?:
?
0?????????

?
1?????????
?
-__inference_brute_force_1_layer_call_fn_17769|44?1
*?'
?
input_1?????????

 
p 
? "=?:
?
0?????????

?
1?????????
?
-__inference_brute_force_1_layer_call_fn_17830|44?1
*?'
?
queries?????????

 
p
? "=?:
?
0?????????

?
1?????????
?
-__inference_brute_force_1_layer_call_fn_17847|44?1
*?'
?
queries?????????

 
p 
? "=?:
?
0?????????

?
1?????????
?
G__inference_embedding_11_layer_call_and_return_conditional_losses_17902W+?(
!?
?
inputs?????????	
? "%?"
?
0????????? 
? z
,__inference_embedding_11_layer_call_fn_17909J+?(
!?
?
inputs?????????	
? "?????????? y
__inference_restore_fn_17951YK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_17943?&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17471q4C?@
9?6
,?)
string_lookup_13_input?????????
p

 
? "%?"
?
0????????? 
? ?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17481q4C?@
9?6
,?)
string_lookup_13_input?????????
p 

 
? "%?"
?
0????????? 
? ?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17859a43?0
)?&
?
inputs?????????
p

 
? "%?"
?
0????????? 
? ?
H__inference_sequential_11_layer_call_and_return_conditional_losses_17871a43?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0????????? 
? ?
-__inference_sequential_11_layer_call_fn_17503d4C?@
9?6
,?)
string_lookup_13_input?????????
p

 
? "?????????? ?
-__inference_sequential_11_layer_call_fn_17524d4C?@
9?6
,?)
string_lookup_13_input?????????
p 

 
? "?????????? ?
-__inference_sequential_11_layer_call_fn_17882T43?0
)?&
?
inputs?????????
p

 
? "?????????? ?
-__inference_sequential_11_layer_call_fn_17893T43?0
)?&
?
inputs?????????
p 

 
? "?????????? ?
#__inference_signature_wrapper_17664?47?4
? 
-?*
(
input_1?
input_1?????????"c?`
.
output_1"?
output_1?????????

.
output_2"?
output_2?????????
