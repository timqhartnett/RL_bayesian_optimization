£­
¾¢
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
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
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02unknown8¬Ó

dqn_45/dense_135/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_namedqn_45/dense_135/kernel

+dqn_45/dense_135/kernel/Read/ReadVariableOpReadVariableOpdqn_45/dense_135/kernel*
_output_shapes

:@*
dtype0

dqn_45/dense_135/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_namedqn_45/dense_135/bias
{
)dqn_45/dense_135/bias/Read/ReadVariableOpReadVariableOpdqn_45/dense_135/bias*
_output_shapes
:@*
dtype0

dqn_45/dense_136/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *(
shared_namedqn_45/dense_136/kernel

+dqn_45/dense_136/kernel/Read/ReadVariableOpReadVariableOpdqn_45/dense_136/kernel*
_output_shapes

:@ *
dtype0

dqn_45/dense_136/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namedqn_45/dense_136/bias
{
)dqn_45/dense_136/bias/Read/ReadVariableOpReadVariableOpdqn_45/dense_136/bias*
_output_shapes
: *
dtype0

dqn_45/dense_137/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_namedqn_45/dense_137/kernel

+dqn_45/dense_137/kernel/Read/ReadVariableOpReadVariableOpdqn_45/dense_137/kernel*
_output_shapes

: *
dtype0

dqn_45/dense_137/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namedqn_45/dense_137/bias
{
)dqn_45/dense_137/bias/Read/ReadVariableOpReadVariableOpdqn_45/dense_137/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dqn_45/dense_135/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name Adam/dqn_45/dense_135/kernel/m

2Adam/dqn_45/dense_135/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_135/kernel/m*
_output_shapes

:@*
dtype0

Adam/dqn_45/dense_135/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/dqn_45/dense_135/bias/m

0Adam/dqn_45/dense_135/bias/m/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_135/bias/m*
_output_shapes
:@*
dtype0

Adam/dqn_45/dense_136/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ */
shared_name Adam/dqn_45/dense_136/kernel/m

2Adam/dqn_45/dense_136/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_136/kernel/m*
_output_shapes

:@ *
dtype0

Adam/dqn_45/dense_136/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/dqn_45/dense_136/bias/m

0Adam/dqn_45/dense_136/bias/m/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_136/bias/m*
_output_shapes
: *
dtype0

Adam/dqn_45/dense_137/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: */
shared_name Adam/dqn_45/dense_137/kernel/m

2Adam/dqn_45/dense_137/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_137/kernel/m*
_output_shapes

: *
dtype0

Adam/dqn_45/dense_137/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/dqn_45/dense_137/bias/m

0Adam/dqn_45/dense_137/bias/m/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_137/bias/m*
_output_shapes
:*
dtype0

Adam/dqn_45/dense_135/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name Adam/dqn_45/dense_135/kernel/v

2Adam/dqn_45/dense_135/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_135/kernel/v*
_output_shapes

:@*
dtype0

Adam/dqn_45/dense_135/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/dqn_45/dense_135/bias/v

0Adam/dqn_45/dense_135/bias/v/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_135/bias/v*
_output_shapes
:@*
dtype0

Adam/dqn_45/dense_136/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ */
shared_name Adam/dqn_45/dense_136/kernel/v

2Adam/dqn_45/dense_136/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_136/kernel/v*
_output_shapes

:@ *
dtype0

Adam/dqn_45/dense_136/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/dqn_45/dense_136/bias/v

0Adam/dqn_45/dense_136/bias/v/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_136/bias/v*
_output_shapes
: *
dtype0

Adam/dqn_45/dense_137/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: */
shared_name Adam/dqn_45/dense_137/kernel/v

2Adam/dqn_45/dense_137/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_137/kernel/v*
_output_shapes

: *
dtype0

Adam/dqn_45/dense_137/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/dqn_45/dense_137/bias/v

0Adam/dqn_45/dense_137/bias/v/Read/ReadVariableOpReadVariableOpAdam/dqn_45/dense_137/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*½(
value³(B°( B©(
å
fc1
fc2
q
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*
°
%iter

&beta_1

'beta_2
	(decay
)learning_ratemFmGmHmImJmKvLvMvNvOvPvQ*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*

*0
+1* 
°
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

1serving_default* 
VP
VARIABLE_VALUEdqn_45/dense_135/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdqn_45/dense_135/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
	
*0* 

2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
VP
VARIABLE_VALUEdqn_45/dense_136/kernel%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdqn_45/dense_136/bias#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
	
+0* 

7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEdqn_45/dense_137/kernel#q/kernel/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdqn_45/dense_137/bias!q/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0
1
2*

A0*
* 
* 
* 
* 
* 
* 
	
*0* 
* 
* 
* 
* 
	
+0* 
* 
* 
* 
* 
* 
* 
8
	Btotal
	Ccount
D	variables
E	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

B0
C1*

D	variables*
ys
VARIABLE_VALUEAdam/dqn_45/dense_135/kernel/mAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/dqn_45/dense_135/bias/m?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/dqn_45/dense_136/kernel/mAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/dqn_45/dense_136/bias/m?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/dqn_45/dense_137/kernel/m?q/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dqn_45/dense_137/bias/m=q/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/dqn_45/dense_135/kernel/vAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/dqn_45/dense_135/bias/v?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/dqn_45/dense_136/kernel/vAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/dqn_45/dense_136/bias/v?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/dqn_45/dense_137/kernel/v?q/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dqn_45/dense_137/bias/v=q/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ð
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dqn_45/dense_135/kerneldqn_45/dense_135/biasdqn_45/dense_136/kerneldqn_45/dense_136/biasdqn_45/dense_137/kerneldqn_45/dense_137/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_53803069
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+dqn_45/dense_135/kernel/Read/ReadVariableOp)dqn_45/dense_135/bias/Read/ReadVariableOp+dqn_45/dense_136/kernel/Read/ReadVariableOp)dqn_45/dense_136/bias/Read/ReadVariableOp+dqn_45/dense_137/kernel/Read/ReadVariableOp)dqn_45/dense_137/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2Adam/dqn_45/dense_135/kernel/m/Read/ReadVariableOp0Adam/dqn_45/dense_135/bias/m/Read/ReadVariableOp2Adam/dqn_45/dense_136/kernel/m/Read/ReadVariableOp0Adam/dqn_45/dense_136/bias/m/Read/ReadVariableOp2Adam/dqn_45/dense_137/kernel/m/Read/ReadVariableOp0Adam/dqn_45/dense_137/bias/m/Read/ReadVariableOp2Adam/dqn_45/dense_135/kernel/v/Read/ReadVariableOp0Adam/dqn_45/dense_135/bias/v/Read/ReadVariableOp2Adam/dqn_45/dense_136/kernel/v/Read/ReadVariableOp0Adam/dqn_45/dense_136/bias/v/Read/ReadVariableOp2Adam/dqn_45/dense_137/kernel/v/Read/ReadVariableOp0Adam/dqn_45/dense_137/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
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
GPU 2J 8 **
f%R#
!__inference__traced_save_53803272

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedqn_45/dense_135/kerneldqn_45/dense_135/biasdqn_45/dense_136/kerneldqn_45/dense_136/biasdqn_45/dense_137/kerneldqn_45/dense_137/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dqn_45/dense_135/kernel/mAdam/dqn_45/dense_135/bias/mAdam/dqn_45/dense_136/kernel/mAdam/dqn_45/dense_136/bias/mAdam/dqn_45/dense_137/kernel/mAdam/dqn_45/dense_137/bias/mAdam/dqn_45/dense_135/kernel/vAdam/dqn_45/dense_135/bias/vAdam/dqn_45/dense_136/kernel/vAdam/dqn_45/dense_136/bias/vAdam/dqn_45/dense_137/kernel/vAdam/dqn_45/dense_137/bias/v*%
Tin
2*
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_53803357µß
ç

)__inference_dqn_45_layer_call_fn_53803014	
state
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dqn_45_layer_call_and_return_conditional_losses_53802886o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
Ê
Ã
__inference_loss_fn_0_53803163T
Bdqn_45_dense_135_kernel_regularizer_square_readvariableop_resource:@
identity¢9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp¼
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBdqn_45_dense_135_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@*
dtype0 
*dqn_45/dense_135/kernel/Regularizer/SquareSquareAdqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@z
)dqn_45/dense_135/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_135/kernel/Regularizer/SumSum.dqn_45/dense_135/kernel/Regularizer/Square:y:02dqn_45/dense_135/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_135/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_135/kernel/Regularizer/mulMul2dqn_45/dense_135/kernel/Regularizer/mul/x:output:00dqn_45/dense_135/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentity+dqn_45/dense_135/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp:^dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp
È

,__inference_dense_136_layer_call_fn_53803116

inputs
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_136_layer_call_and_return_conditional_losses_53802851o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§
´
G__inference_dense_136_layer_call_and_return_conditional_losses_53802851

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0 
*dqn_45/dense_136/kernel/Regularizer/SquareSquareAdqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ z
)dqn_45/dense_136/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_136/kernel/Regularizer/SumSum.dqn_45/dense_136/kernel/Regularizer/Square:y:02dqn_45/dense_136/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_136/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_136/kernel/Regularizer/mulMul2dqn_45/dense_136/kernel/Regularizer/mul/x:output:00dqn_45/dense_136/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp:^dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2v
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Å
#__inference__wrapped_model_53802804
input_1A
/dqn_45_dense_135_matmul_readvariableop_resource:@>
0dqn_45_dense_135_biasadd_readvariableop_resource:@A
/dqn_45_dense_136_matmul_readvariableop_resource:@ >
0dqn_45_dense_136_biasadd_readvariableop_resource: A
/dqn_45_dense_137_matmul_readvariableop_resource: >
0dqn_45_dense_137_biasadd_readvariableop_resource:
identity¢'dqn_45/dense_135/BiasAdd/ReadVariableOp¢&dqn_45/dense_135/MatMul/ReadVariableOp¢'dqn_45/dense_136/BiasAdd/ReadVariableOp¢&dqn_45/dense_136/MatMul/ReadVariableOp¢'dqn_45/dense_137/BiasAdd/ReadVariableOp¢&dqn_45/dense_137/MatMul/ReadVariableOp
&dqn_45/dense_135/MatMul/ReadVariableOpReadVariableOp/dqn_45_dense_135_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dqn_45/dense_135/MatMulMatMulinput_1.dqn_45/dense_135/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'dqn_45/dense_135/BiasAdd/ReadVariableOpReadVariableOp0dqn_45_dense_135_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
dqn_45/dense_135/BiasAddBiasAdd!dqn_45/dense_135/MatMul:product:0/dqn_45/dense_135/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
dqn_45/dense_135/ReluRelu!dqn_45/dense_135/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&dqn_45/dense_136/MatMul/ReadVariableOpReadVariableOp/dqn_45_dense_136_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0¨
dqn_45/dense_136/MatMulMatMul#dqn_45/dense_135/Relu:activations:0.dqn_45/dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'dqn_45/dense_136/BiasAdd/ReadVariableOpReadVariableOp0dqn_45_dense_136_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
dqn_45/dense_136/BiasAddBiasAdd!dqn_45/dense_136/MatMul:product:0/dqn_45/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
dqn_45/dense_136/ReluRelu!dqn_45/dense_136/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&dqn_45/dense_137/MatMul/ReadVariableOpReadVariableOp/dqn_45_dense_137_matmul_readvariableop_resource*
_output_shapes

: *
dtype0¨
dqn_45/dense_137/MatMulMatMul#dqn_45/dense_136/Relu:activations:0.dqn_45/dense_137/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'dqn_45/dense_137/BiasAdd/ReadVariableOpReadVariableOp0dqn_45_dense_137_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
dqn_45/dense_137/BiasAddBiasAdd!dqn_45/dense_137/MatMul:product:0/dqn_45/dense_137/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
IdentityIdentity!dqn_45/dense_137/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
NoOpNoOp(^dqn_45/dense_135/BiasAdd/ReadVariableOp'^dqn_45/dense_135/MatMul/ReadVariableOp(^dqn_45/dense_136/BiasAdd/ReadVariableOp'^dqn_45/dense_136/MatMul/ReadVariableOp(^dqn_45/dense_137/BiasAdd/ReadVariableOp'^dqn_45/dense_137/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2R
'dqn_45/dense_135/BiasAdd/ReadVariableOp'dqn_45/dense_135/BiasAdd/ReadVariableOp2P
&dqn_45/dense_135/MatMul/ReadVariableOp&dqn_45/dense_135/MatMul/ReadVariableOp2R
'dqn_45/dense_136/BiasAdd/ReadVariableOp'dqn_45/dense_136/BiasAdd/ReadVariableOp2P
&dqn_45/dense_136/MatMul/ReadVariableOp&dqn_45/dense_136/MatMul/ReadVariableOp2R
'dqn_45/dense_137/BiasAdd/ReadVariableOp'dqn_45/dense_137/BiasAdd/ReadVariableOp2P
&dqn_45/dense_137/MatMul/ReadVariableOp&dqn_45/dense_137/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ê
Ã
__inference_loss_fn_1_53803174T
Bdqn_45_dense_136_kernel_regularizer_square_readvariableop_resource:@ 
identity¢9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp¼
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBdqn_45_dense_136_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@ *
dtype0 
*dqn_45/dense_136/kernel/Regularizer/SquareSquareAdqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ z
)dqn_45/dense_136/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_136/kernel/Regularizer/SumSum.dqn_45/dense_136/kernel/Regularizer/Square:y:02dqn_45/dense_136/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_136/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_136/kernel/Regularizer/mulMul2dqn_45/dense_136/kernel/Regularizer/mul/x:output:00dqn_45/dense_136/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentity+dqn_45/dense_136/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp:^dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp
È

,__inference_dense_137_layer_call_fn_53803142

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_137_layer_call_and_return_conditional_losses_53802867o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¤e

$__inference__traced_restore_53803357
file_prefix:
(assignvariableop_dqn_45_dense_135_kernel:@6
(assignvariableop_1_dqn_45_dense_135_bias:@<
*assignvariableop_2_dqn_45_dense_136_kernel:@ 6
(assignvariableop_3_dqn_45_dense_136_bias: <
*assignvariableop_4_dqn_45_dense_137_kernel: 6
(assignvariableop_5_dqn_45_dense_137_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: D
2assignvariableop_13_adam_dqn_45_dense_135_kernel_m:@>
0assignvariableop_14_adam_dqn_45_dense_135_bias_m:@D
2assignvariableop_15_adam_dqn_45_dense_136_kernel_m:@ >
0assignvariableop_16_adam_dqn_45_dense_136_bias_m: D
2assignvariableop_17_adam_dqn_45_dense_137_kernel_m: >
0assignvariableop_18_adam_dqn_45_dense_137_bias_m:D
2assignvariableop_19_adam_dqn_45_dense_135_kernel_v:@>
0assignvariableop_20_adam_dqn_45_dense_135_bias_v:@D
2assignvariableop_21_adam_dqn_45_dense_136_kernel_v:@ >
0assignvariableop_22_adam_dqn_45_dense_136_bias_v: D
2assignvariableop_23_adam_dqn_45_dense_137_kernel_v: >
0assignvariableop_24_adam_dqn_45_dense_137_bias_v:
identity_26¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Î
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ô

valueê
Bç
B%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB#q/kernel/.ATTRIBUTES/VARIABLE_VALUEB!q/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?q/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=q/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?q/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=q/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B  
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp(assignvariableop_dqn_45_dense_135_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp(assignvariableop_1_dqn_45_dense_135_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp*assignvariableop_2_dqn_45_dense_136_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp(assignvariableop_3_dqn_45_dense_136_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp*assignvariableop_4_dqn_45_dense_137_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp(assignvariableop_5_dqn_45_dense_137_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_13AssignVariableOp2assignvariableop_13_adam_dqn_45_dense_135_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_14AssignVariableOp0assignvariableop_14_adam_dqn_45_dense_135_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_15AssignVariableOp2assignvariableop_15_adam_dqn_45_dense_136_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_16AssignVariableOp0assignvariableop_16_adam_dqn_45_dense_136_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_dqn_45_dense_137_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_dqn_45_dense_137_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_dqn_45_dense_135_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_dqn_45_dense_135_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_dqn_45_dense_136_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_dqn_45_dense_136_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_dqn_45_dense_137_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_dqn_45_dense_137_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 õ
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: â
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
´$

D__inference_dqn_45_layer_call_and_return_conditional_losses_53802886	
state$
dense_135_53802829:@ 
dense_135_53802831:@$
dense_136_53802852:@  
dense_136_53802854: $
dense_137_53802868:  
dense_137_53802870:
identity¢!dense_135/StatefulPartitionedCall¢!dense_136/StatefulPartitionedCall¢!dense_137/StatefulPartitionedCall¢9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp¢9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOpù
!dense_135/StatefulPartitionedCallStatefulPartitionedCallstatedense_135_53802829dense_135_53802831*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_135_layer_call_and_return_conditional_losses_53802828
!dense_136/StatefulPartitionedCallStatefulPartitionedCall*dense_135/StatefulPartitionedCall:output:0dense_136_53802852dense_136_53802854*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_136_layer_call_and_return_conditional_losses_53802851
!dense_137/StatefulPartitionedCallStatefulPartitionedCall*dense_136/StatefulPartitionedCall:output:0dense_137_53802868dense_137_53802870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_137_layer_call_and_return_conditional_losses_53802867
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_135_53802829*
_output_shapes

:@*
dtype0 
*dqn_45/dense_135/kernel/Regularizer/SquareSquareAdqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@z
)dqn_45/dense_135/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_135/kernel/Regularizer/SumSum.dqn_45/dense_135/kernel/Regularizer/Square:y:02dqn_45/dense_135/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_135/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_135/kernel/Regularizer/mulMul2dqn_45/dense_135/kernel/Regularizer/mul/x:output:00dqn_45/dense_135/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_136_53802852*
_output_shapes

:@ *
dtype0 
*dqn_45/dense_136/kernel/Regularizer/SquareSquareAdqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ z
)dqn_45/dense_136/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_136/kernel/Regularizer/SumSum.dqn_45/dense_136/kernel/Regularizer/Square:y:02dqn_45/dense_136/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_136/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_136/kernel/Regularizer/mulMul2dqn_45/dense_136/kernel/Regularizer/mul/x:output:00dqn_45/dense_136/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_137/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp"^dense_135/StatefulPartitionedCall"^dense_136/StatefulPartitionedCall"^dense_137/StatefulPartitionedCall:^dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:^dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dense_135/StatefulPartitionedCall!dense_135/StatefulPartitionedCall2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2F
!dense_137/StatefulPartitionedCall!dense_137/StatefulPartitionedCall2v
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp2v
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
È

,__inference_dense_135_layer_call_fn_53803084

inputs
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_135_layer_call_and_return_conditional_losses_53802828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
´
G__inference_dense_136_layer_call_and_return_conditional_losses_53803133

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0 
*dqn_45/dense_136/kernel/Regularizer/SquareSquareAdqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ z
)dqn_45/dense_136/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_136/kernel/Regularizer/SumSum.dqn_45/dense_136/kernel/Regularizer/Square:y:02dqn_45/dense_136/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_136/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_136/kernel/Regularizer/mulMul2dqn_45/dense_136/kernel/Regularizer/mul/x:output:00dqn_45/dense_136/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp:^dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2v
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
í

)__inference_dqn_45_layer_call_fn_53802901
input_1
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dqn_45_layer_call_and_return_conditional_losses_53802886o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ê	
ø
G__inference_dense_137_layer_call_and_return_conditional_losses_53803152

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_137_layer_call_and_return_conditional_losses_53802867

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ô8
Ì
!__inference__traced_save_53803272
file_prefix6
2savev2_dqn_45_dense_135_kernel_read_readvariableop4
0savev2_dqn_45_dense_135_bias_read_readvariableop6
2savev2_dqn_45_dense_136_kernel_read_readvariableop4
0savev2_dqn_45_dense_136_bias_read_readvariableop6
2savev2_dqn_45_dense_137_kernel_read_readvariableop4
0savev2_dqn_45_dense_137_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_adam_dqn_45_dense_135_kernel_m_read_readvariableop;
7savev2_adam_dqn_45_dense_135_bias_m_read_readvariableop=
9savev2_adam_dqn_45_dense_136_kernel_m_read_readvariableop;
7savev2_adam_dqn_45_dense_136_bias_m_read_readvariableop=
9savev2_adam_dqn_45_dense_137_kernel_m_read_readvariableop;
7savev2_adam_dqn_45_dense_137_bias_m_read_readvariableop=
9savev2_adam_dqn_45_dense_135_kernel_v_read_readvariableop;
7savev2_adam_dqn_45_dense_135_bias_v_read_readvariableop=
9savev2_adam_dqn_45_dense_136_kernel_v_read_readvariableop;
7savev2_adam_dqn_45_dense_136_bias_v_read_readvariableop=
9savev2_adam_dqn_45_dense_137_kernel_v_read_readvariableop;
7savev2_adam_dqn_45_dense_137_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ë
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ô

valueê
Bç
B%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB#q/kernel/.ATTRIBUTES/VARIABLE_VALUEB!q/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?q/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB=q/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?q/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB=q/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¡
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B Ã
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_dqn_45_dense_135_kernel_read_readvariableop0savev2_dqn_45_dense_135_bias_read_readvariableop2savev2_dqn_45_dense_136_kernel_read_readvariableop0savev2_dqn_45_dense_136_bias_read_readvariableop2savev2_dqn_45_dense_137_kernel_read_readvariableop0savev2_dqn_45_dense_137_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_adam_dqn_45_dense_135_kernel_m_read_readvariableop7savev2_adam_dqn_45_dense_135_bias_m_read_readvariableop9savev2_adam_dqn_45_dense_136_kernel_m_read_readvariableop7savev2_adam_dqn_45_dense_136_bias_m_read_readvariableop9savev2_adam_dqn_45_dense_137_kernel_m_read_readvariableop7savev2_adam_dqn_45_dense_137_bias_m_read_readvariableop9savev2_adam_dqn_45_dense_135_kernel_v_read_readvariableop7savev2_adam_dqn_45_dense_135_bias_v_read_readvariableop9savev2_adam_dqn_45_dense_136_kernel_v_read_readvariableop7savev2_adam_dqn_45_dense_136_bias_v_read_readvariableop9savev2_adam_dqn_45_dense_137_kernel_v_read_readvariableop7savev2_adam_dqn_45_dense_137_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*·
_input_shapes¥
¢: :@:@:@ : : :: : : : : : : :@:@:@ : : ::@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
º$
 
D__inference_dqn_45_layer_call_and_return_conditional_losses_53802979
input_1$
dense_135_53802951:@ 
dense_135_53802953:@$
dense_136_53802956:@  
dense_136_53802958: $
dense_137_53802961:  
dense_137_53802963:
identity¢!dense_135/StatefulPartitionedCall¢!dense_136/StatefulPartitionedCall¢!dense_137/StatefulPartitionedCall¢9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp¢9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOpû
!dense_135/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_135_53802951dense_135_53802953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_135_layer_call_and_return_conditional_losses_53802828
!dense_136/StatefulPartitionedCallStatefulPartitionedCall*dense_135/StatefulPartitionedCall:output:0dense_136_53802956dense_136_53802958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_136_layer_call_and_return_conditional_losses_53802851
!dense_137/StatefulPartitionedCallStatefulPartitionedCall*dense_136/StatefulPartitionedCall:output:0dense_137_53802961dense_137_53802963*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_137_layer_call_and_return_conditional_losses_53802867
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_135_53802951*
_output_shapes

:@*
dtype0 
*dqn_45/dense_135/kernel/Regularizer/SquareSquareAdqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@z
)dqn_45/dense_135/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_135/kernel/Regularizer/SumSum.dqn_45/dense_135/kernel/Regularizer/Square:y:02dqn_45/dense_135/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_135/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_135/kernel/Regularizer/mulMul2dqn_45/dense_135/kernel/Regularizer/mul/x:output:00dqn_45/dense_135/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_136_53802956*
_output_shapes

:@ *
dtype0 
*dqn_45/dense_136/kernel/Regularizer/SquareSquareAdqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ z
)dqn_45/dense_136/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_136/kernel/Regularizer/SumSum.dqn_45/dense_136/kernel/Regularizer/Square:y:02dqn_45/dense_136/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_136/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_136/kernel/Regularizer/mulMul2dqn_45/dense_136/kernel/Regularizer/mul/x:output:00dqn_45/dense_136/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_137/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp"^dense_135/StatefulPartitionedCall"^dense_136/StatefulPartitionedCall"^dense_137/StatefulPartitionedCall:^dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:^dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dense_135/StatefulPartitionedCall!dense_135/StatefulPartitionedCall2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2F
!dense_137/StatefulPartitionedCall!dense_137/StatefulPartitionedCall2v
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp2v
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
§
´
G__inference_dense_135_layer_call_and_return_conditional_losses_53803101

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0 
*dqn_45/dense_135/kernel/Regularizer/SquareSquareAdqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@z
)dqn_45/dense_135/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_135/kernel/Regularizer/SumSum.dqn_45/dense_135/kernel/Regularizer/Square:y:02dqn_45/dense_135/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_135/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_135/kernel/Regularizer/mulMul2dqn_45/dense_135/kernel/Regularizer/mul/x:output:00dqn_45/dense_135/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp:^dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2v
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
´
G__inference_dense_135_layer_call_and_return_conditional_losses_53802828

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0 
*dqn_45/dense_135/kernel/Regularizer/SquareSquareAdqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@z
)dqn_45/dense_135/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_135/kernel/Regularizer/SumSum.dqn_45/dense_135/kernel/Regularizer/Square:y:02dqn_45/dense_135/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_135/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_135/kernel/Regularizer/mulMul2dqn_45/dense_135/kernel/Regularizer/mul/x:output:00dqn_45/dense_135/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp:^dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2v
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

&__inference_signature_wrapper_53803069
input_1
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_53802804o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
.

D__inference_dqn_45_layer_call_and_return_conditional_losses_53803050	
state:
(dense_135_matmul_readvariableop_resource:@7
)dense_135_biasadd_readvariableop_resource:@:
(dense_136_matmul_readvariableop_resource:@ 7
)dense_136_biasadd_readvariableop_resource: :
(dense_137_matmul_readvariableop_resource: 7
)dense_137_biasadd_readvariableop_resource:
identity¢ dense_135/BiasAdd/ReadVariableOp¢dense_135/MatMul/ReadVariableOp¢ dense_136/BiasAdd/ReadVariableOp¢dense_136/MatMul/ReadVariableOp¢ dense_137/BiasAdd/ReadVariableOp¢dense_137/MatMul/ReadVariableOp¢9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp¢9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp
dense_135/MatMul/ReadVariableOpReadVariableOp(dense_135_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0|
dense_135/MatMulMatMulstate'dense_135/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 dense_135/BiasAdd/ReadVariableOpReadVariableOp)dense_135_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_135/BiasAddBiasAdddense_135/MatMul:product:0(dense_135/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
dense_135/ReluReludense_135/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_136/MatMul/ReadVariableOpReadVariableOp(dense_136_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_136/MatMulMatMuldense_135/Relu:activations:0'dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 dense_136/BiasAdd/ReadVariableOpReadVariableOp)dense_136_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_136/BiasAddBiasAdddense_136/MatMul:product:0(dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
dense_136/ReluReludense_136/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_137/MatMul/ReadVariableOpReadVariableOp(dense_137_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_137/MatMulMatMuldense_136/Relu:activations:0'dense_137/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_137/BiasAdd/ReadVariableOpReadVariableOp)dense_137_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_137/BiasAddBiasAdddense_137/MatMul:product:0(dense_137/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_135_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0 
*dqn_45/dense_135/kernel/Regularizer/SquareSquareAdqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@z
)dqn_45/dense_135/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_135/kernel/Regularizer/SumSum.dqn_45/dense_135/kernel/Regularizer/Square:y:02dqn_45/dense_135/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_135/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_135/kernel/Regularizer/mulMul2dqn_45/dense_135/kernel/Regularizer/mul/x:output:00dqn_45/dense_135/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_136_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0 
*dqn_45/dense_136/kernel/Regularizer/SquareSquareAdqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ z
)dqn_45/dense_136/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ³
'dqn_45/dense_136/kernel/Regularizer/SumSum.dqn_45/dense_136/kernel/Regularizer/Square:y:02dqn_45/dense_136/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: n
)dqn_45/dense_136/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
'dqn_45/dense_136/kernel/Regularizer/mulMul2dqn_45/dense_136/kernel/Regularizer/mul/x:output:00dqn_45/dense_136/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_137/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_135/BiasAdd/ReadVariableOp ^dense_135/MatMul/ReadVariableOp!^dense_136/BiasAdd/ReadVariableOp ^dense_136/MatMul/ReadVariableOp!^dense_137/BiasAdd/ReadVariableOp ^dense_137/MatMul/ReadVariableOp:^dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp:^dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2D
 dense_135/BiasAdd/ReadVariableOp dense_135/BiasAdd/ReadVariableOp2B
dense_135/MatMul/ReadVariableOpdense_135/MatMul/ReadVariableOp2D
 dense_136/BiasAdd/ReadVariableOp dense_136/BiasAdd/ReadVariableOp2B
dense_136/MatMul/ReadVariableOpdense_136/MatMul/ReadVariableOp2D
 dense_137/BiasAdd/ReadVariableOp dense_137/BiasAdd/ReadVariableOp2B
dense_137/MatMul/ReadVariableOpdense_137/MatMul/ReadVariableOp2v
9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_135/kernel/Regularizer/Square/ReadVariableOp2v
9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp9dqn_45/dense_136/kernel/Regularizer/Square/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÒI
ú
fc1
fc2
q
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
¿
%iter

&beta_1

'beta_2
	(decay
)learning_ratemFmGmHmImJmKvLvMvNvOvPvQ"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
Ê
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
ý2ú
)__inference_dqn_45_layer_call_fn_53802901
)__inference_dqn_45_layer_call_fn_53803014¡
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³2°
D__inference_dqn_45_layer_call_and_return_conditional_losses_53803050
D__inference_dqn_45_layer_call_and_return_conditional_losses_53802979¡
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÎBË
#__inference__wrapped_model_53802804input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
1serving_default"
signature_map
):'@2dqn_45/dense_135/kernel
#:!@2dqn_45/dense_135/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
­
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_135_layer_call_fn_53803084¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_135_layer_call_and_return_conditional_losses_53803101¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
):'@ 2dqn_45/dense_136/kernel
#:! 2dqn_45/dense_136/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
­
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_136_layer_call_fn_53803116¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_136_layer_call_and_return_conditional_losses_53803133¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
):' 2dqn_45/dense_137/kernel
#:!2dqn_45/dense_137/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_137_layer_call_fn_53803142¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_137_layer_call_and_return_conditional_losses_53803152¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
µ2²
__inference_loss_fn_0_53803163
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_1_53803174
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÍBÊ
&__inference_signature_wrapper_53803069input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
+0"
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
N
	Btotal
	Ccount
D	variables
E	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
B0
C1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
.:,@2Adam/dqn_45/dense_135/kernel/m
(:&@2Adam/dqn_45/dense_135/bias/m
.:,@ 2Adam/dqn_45/dense_136/kernel/m
(:& 2Adam/dqn_45/dense_136/bias/m
.:, 2Adam/dqn_45/dense_137/kernel/m
(:&2Adam/dqn_45/dense_137/bias/m
.:,@2Adam/dqn_45/dense_135/kernel/v
(:&@2Adam/dqn_45/dense_135/bias/v
.:,@ 2Adam/dqn_45/dense_136/kernel/v
(:& 2Adam/dqn_45/dense_136/bias/v
.:, 2Adam/dqn_45/dense_137/kernel/v
(:&2Adam/dqn_45/dense_137/bias/v
#__inference__wrapped_model_53802804o0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_135_layer_call_and_return_conditional_losses_53803101\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_dense_135_layer_call_fn_53803084O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@§
G__inference_dense_136_layer_call_and_return_conditional_losses_53803133\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_dense_136_layer_call_fn_53803116O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ §
G__inference_dense_137_layer_call_and_return_conditional_losses_53803152\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_137_layer_call_fn_53803142O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ©
D__inference_dqn_45_layer_call_and_return_conditional_losses_53802979a0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 §
D__inference_dqn_45_layer_call_and_return_conditional_losses_53803050_.¢+
$¢!

stateÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_dqn_45_layer_call_fn_53802901T0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_dqn_45_layer_call_fn_53803014R.¢+
$¢!

stateÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ=
__inference_loss_fn_0_53803163¢

¢ 
ª " =
__inference_loss_fn_1_53803174¢

¢ 
ª " ¤
&__inference_signature_wrapper_53803069z;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ