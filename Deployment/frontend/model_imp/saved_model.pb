��)
�'�'
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
=
Greater
x"T
y"T
z
"
Ttype:
2	
�
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�
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
Tvaluestype�
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype�
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype�
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
�
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
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
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint���������
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.22v2.9.1-132-g18960c44ad38��(
�
Nadam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_3/bias/v
y
(Nadam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_3/bias/v*
_output_shapes
:*
dtype0
�
Nadam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameNadam/dense_3/kernel/v
�
*Nadam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_3/kernel/v*
_output_shapes

: *
dtype0
�
Nadam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/dense_2/bias/v
y
(Nadam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_2/bias/v*
_output_shapes
: *
dtype0
�
Nadam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *'
shared_nameNadam/dense_2/kernel/v
�
*Nadam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_2/kernel/v*
_output_shapes
:	� *
dtype0
�
Nadam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*-
shared_nameNadam/embedding/embeddings/v
�
0Nadam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpNadam/embedding/embeddings/v*!
_output_shapes
:���*
dtype0
�
Nadam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_3/bias/m
y
(Nadam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_3/bias/m*
_output_shapes
:*
dtype0
�
Nadam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameNadam/dense_3/kernel/m
�
*Nadam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_3/kernel/m*
_output_shapes

: *
dtype0
�
Nadam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/dense_2/bias/m
y
(Nadam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_2/bias/m*
_output_shapes
: *
dtype0
�
Nadam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *'
shared_nameNadam/dense_2/kernel/m
�
*Nadam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_2/kernel/m*
_output_shapes
:	� *
dtype0
�
Nadam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*-
shared_nameNadam/embedding/embeddings/m
�
0Nadam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpNadam/embedding/embeddings/m*!
_output_shapes
:���*
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	
l

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name4490*
value_dtype0	
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	� *
dtype0
�
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*%
shared_nameembedding/embeddings
�
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*!
_output_shapes
:���*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
��
Const_4Const*
_output_shapes

:��*
dtype0*��
value��B����BsesuaiBbarangBbagusBdanBpesananBdiBcepatBnyaBygBsampaiBdenganBsayaBtidakBhargaBbaikBtapiBpackingBsudahBadaBgambarBokBmantapBlumayanBresponB	barangnyaB
pengirimanByangBterimaBwarnaBkualitasBtpBgaBbisaBganBsukaBkurangBlamaBbahanBbangetBlahByaBokeBkasihBsellerBmurahBsangatBiniBpasBgakBsemogaBkerenBsamaBjugaBbuatBuntukBrapiBukuranBlebihBdgnBlagiBagakBbeliBcumaB	deskripsiBgoodBmakasihBajaBdikirimBterimakasihBpuasBdBthanksBpesanBawetBrapihBditerimaBkecewaBtdkBdariB	berfungsiBbgtBkecilBbaruBiBbelumBpelapakBjadiBcobaBkeBsekaliBcepetBdehBdahBpakaiBsampeBbahannyaBudahBchatBpakeBkirimBsepeBsyBtipisBcocokBpokoknyaBmauBsdhBorderBsuksesBfotoBmudahBcukupBhanyaBgkBrecommendedBdipakaiBsihBternyataBharganyaBhariB
recomendedBnBbiasaBdatangBbedaBapaB	bukalapakBkarenaB	memuaskanBbanyakBenakBnyamanBdgBsajaBdicobaBsayangBlainBprodukBjgBrusakBhitamBsemuaBbrgByBsipBtopBterlaluBpenjualBkaloBsepatuBmasihBbintangBblmBbukanBpadahalBsedikitBamanBflashBjdBmengecewakanBlgBtpiBaposBakanBanBlapakBjahitanBkalauBmalahB	pelayananBlangsungBnamunBthxBmintaBwaktuBharapanBkloBdalamBanakBituBsizeBkokBbiarBjelekBfastBademBtasBkaliBnoBcumanBsatuBterusBmodelBmungkinBpengirimannyaBmantabBcptBnyBpaketBramahBbukaBselaluBbagianBpesenBharusBsegituButkBwarnanyaBnyampeBsuaraB	ukurannyaBatauBprosesBdapatBbesarBpakingBsetelahBudhBpadaBgBakuBduluBbelanjaBwalaupunBaBkondisiB	sepatunyaBkayaBlambatBcelanaBsalahBtolongBjamBdrBselamatBtebalBpeBcmBalhamdulillahBtepatBbosBdipakeB	sampainyaBsaatB
packingnyaBnormalBbgusBkarnaBtrimsBamaBhpBasliBkakiBoverallB	kekecilanBdapetBtauBputihBmurahanBnyeselBkwalitasBtksBkualitasnyaBpembeliBkrnBjneBjanganBcacatB
bermanfaatBpanjangBcekBtetapiBdisiniBbajuBbsBbiruBdptBbrangBatasBmohonBmkshBwalauBdealByahBpunBbonusBsaleBkanBtokoBgppB
trimakasihBpokonyaBmerahBhasilBhalusBiklanBlBmemangBjauhBnihB
terjangkauBmodelnyaBsmBpanasBdigambarBsyaB	keinginanBjelasBlancarBdikitBpesenanBsBsebelumBkuatByaaBkemasanBthankBsmogaB	digunakanBblBnyalaBtakBnggaBmuatBpcsBoriBteBmaafBdikasihBsiBmahBtanpaBnextBmerkBhasilnyaBsuaranyaBxlBnggakBmBbagussBlemBtqBsmaBkurirBtrimaBmogaBnantiBtBberkualitasBplastikB
permintaanBmasalahBdatengBringanBheheBdipesanBbajunyaBkirainBmudahanBisiBxBtasnyaBbgsBapabilaBtaliBpunyaBrbBmulusBgituB
lumayanlahBlbhBjikaBberbedaBlengkapBlembutB
konfirmasiBmatiBuBkaosBkakBkabelBabuBtetapBsenangBtvBbenerBsesuiBpokokBpinkBpenyokBpastiBoriginalBkainBsmpaiBkaBfungsiBgedeBbikinBpentingBpelayanannyaBpasangBagarBtglBsuperBjualBkirimanBkoBkenapaBkatanyaBbekasBjBmeskipunBjosBtujuanBpecahBlepasBkiraBdlmBcaraBsusahBorderanBmahalBluarBtelahBpatahBmingguBmantepBbngtBtombolBsoalnyaB
ekspektasiBduaBuntungBmeriahBkayakBkBgantiBaneBagaBtambahBpictBparahB	berbicaraBtahanBjgnBdtgBtelatBstandarBbrngBaganBjujurBhatiBdusBbenarBaslinyaBsimpleBsendiriBsemuanyaBniceBtinggalBbateraiBpulaBtrusBsiniBjossBmiripBsehariB	sayangnyaBkulitBkitaB
tingkatkanBsehinggaBlucuBkameraBistriB	gambarnyaBallBtanganBrumahB
rekomendedBmantappBtempatBkwBharusnyaBemangB	pelangganBkardusBdoangBpalingBorangBmakinBjobBboxBperluBketikaBbestBbegituBtestB	responnyaBongkirBkonsumenBcoklatB
jahitannyaBitBgampangBehBhrgaBbilangBbatreBokelahBbeberapaBtibaBkomplainB
diharapkanBbawahBsoBsempitBpernahBpendekBmanaBtesBsippBsiipB
keteranganBisinyaBinginkanBakhirnyaBmendaratBbadanBliatBdibukaBrasaBraguBabisBmasaBlgiB	diskripsiBrealBhabisByesBterlihatBsebelahBmshBkamiBjernihBjalanBdnBseginiBseBjgaBtelitiBblomBtebelBsobekBmasukB	langgananBmotifB	kebesaranBkasarBcopotBbonusnyaB	recommendBbeBseringBmantapppB	celananyaBadBrecomendBkembaliBjasaB	flashdealBbahanyaBbagusssBuangBsempatBhrgBberatBniBkiriBjdiBhampirBpromoBonlineBlecetBlampuBkeliatanBkeadaanBkananBjiwaBgunakanBandBalamatByouBtetepBpilihBonBdepanBbungkusBklBreadyBheheheBbossBasalBlcdBkerasBkaretBwangiBsekarangBolehBgimanaBtheBsmpeBsaranBmeskiBvideoBmarkotopBkegedeanBinBbubbleBbiasanyaBbelomBtulisanBreturBpkeB
pelapaknyaBdriBssuaiBjayaBdifotoBcantikBbingungB
seharusnyaBsecaraBkantongBeraBbroBblumBbauBrugiBnotBpihakBpicBmembantuBkerjaBjualanBdipasangBbayarBbagiBulasanBkuningBkuBkendalaBjawabBharapkanBestimasiB	datangnyaBcakepBburukBberkahBpdhlBdijualBlihatBinyaBhijauBbentukBbacaBwanginyaBtinggiB	pembelianBmakasiBkainnyaBbolehBrekomendasiBlikeBwrnaB	nyampenyaBlebarB	kelihatanBgwBalatBahankanB	produknyaBoverB	bluetoothBsisBmenggunakanBheadsetBdechBalangkahBsptBbngetBterbaikBsiangBsemingguBpalsuBomBngakBfullBchargerBcariBsampenyaBperbaikiBbakalBmainBkeluarB
kedepannyaBbukuBantenaBtalinyaBradaBmasBlensaBtanggalBsenengBrasanyaBpraktisBlaptopBlahhBinginBcustomerBbutBajBqualityBnungguBlainnyaBhB
dikirimnyaBresponseBresiBpinggangBpilihanBkataBginiBdluBcpetBchargeBbassBandaBtoBmksihBkotorBkeduaBitemBbatuBawalnyaByaaaBwrapB
sebelumnyaB	responsifB	resletingBputusBpekingBmenarikBkakuBukBspeakerBrequestB	pemakaianBditingkatkanBcasBbergunaBawalBtulisBpengenBmudaBlenganBkotakBwoBstokBslaluBmajuBbrBbicaraBsmpBribetBdongBbotolBbocorB	bentuknyaBbaiknyaBnavyBmerekBlewatBlayarBkosongBgelapB	ekspedisiBapalagiBrobekBresponsBpokokeBgmbarBdngnBdetailBbsaB
selebihnyaB	pngirimanBpassBmelarBkemarinBkapokBdicekBcreamBcmaBudaBsempurnaBrekomenBmayanB	mantappppBjaketBhilangBfreeBeBdompetB
diperbaikiBdhBcpatBbesokBviaBsimpelBnomorBusbBspeB	pemesananBmembeliBmalamBkgBhidupBcmnBbiarkanBbangBaliasBsolBminiBlupaBkirimnyaBkapanBkancingBdnganB	transaksiBterangBsmuaBsgtBmurmerBlapaknyaBkshBkrnaBinfoBfarBbrgnyaB
bersahabatBtungguBseauaiBpdB	packagingB	kecepatanBdahuluBapakahBsoreBringBokeeBmenurutBkonekBkeseluruhanBbilaBambilBrespondBnomerBmainanBlonggarBkrgBjntBjaitanBaplikasiBairBvBtahuB	sellernyaBngaretBnamanyaBlemnyaBharapB	flashsaleBenggaBdtngBbelakangBbekerjaBasBtakutBsupayaBsiapBproductBpayahBpaketnyaBngeBmlBmkasihBmendingBjosssBgratisBdiperhatikanB
diinginkanBdigantiBbadBaqBtutupBtanyaBpotoBpetunjukBlogoBdiskonBxxlBwifiBtopiBremoteBplusBmantafBkokohBkenaBditambahBbersihBsegelBrambutBpowerBmaBjaketnyaBitsBhrsBhancurBgbrBdimintaBbBtuaB	prosesnyaBkykBkoqBkepakeBdngBcatatanBunikBsblmBrandomB
penjualnyaBkerennB	indonesiaB
dipakainyaBdiluarBberhasilB	terlambatB	perkiraanBpagiBpaBiyaBgxBdiaBdaBbykBbenangBbagussssBterasaBtampilanBtajamBsipppBsdBjadinyaBisBgbB
ditawarkanBbelinyaBanehBamatBtrsB	sesuailahB
pesanannyaBpakainyaBoffB	lanjutkanB	kesalahanBkemanaBkadangBimutBguaBgmbrB	fungsinyaBentahBbubleBunguBtrnyataBtaunyaBsempetBoBnamaBmerasaBlebaranBlarisBhahahaBdeskripsinyaBconnectBbodyBbalasBwajarBtimeBsoalBsmgBsistBsakitBrupaBpersisB
kemasannyaBfotonyaBbutuhBbawahnyaBbawaBbangettBbaguslahBadalahBslowBpolosBpisanBngkBminusBkayaknyaBjamnyaBhalBgaransiBentengBbulanBbliBusahBtransferBspekBsatunyaBpsananBpengirimanyaB	packinganBmpBmiringBmantaapBmantaaapBknpBkagaBhrBgueBgorengBcameraBbolongBberjalanBbeneranBalhasilBvolumeBtuhBttpBsukaaBsiiipBsebagaiBorgBlubangBlicinBkencengBfeedbackBeleganBdllBbnyakBbknBbalikBajahBvrBtukarBterlebihBspBseninBsekolahBsedangBsalamBokeyBnyampaiBmenungguBmenjadiBmenitBkilatBkayanyaBhahaBfitBbolaBanggungBwrnBsicepatBresellerBnyampekBniatBmesinBledB	kardusnyaBkacamataBjudulBbarngBallahBtdakBtanggungBtanggapBpapaBmntapBmembuatBkasiBgoldBgmnBditesBulisBukuranyaBsukaaaB	sebaiknyaBrataBpikirBpBnuhunBndakBnaBmataBkrenBkotaBjeansBforBdisplayB	dibungkusBbhnBabalBulangBujungBstockBsablonBrBposBokehBmesenBmanualBmaksimalBlowBkyBkurirnyaBkereenBkedepanBistimewaBdiresponBbuangBbnykBberharapBbahkanBalhamdulilahBahBagkBunitBtrmkshBtipuBsygBstandardBselerBselainBpemulaBpakBmdhBmasukanBlunturBlayakBlaluBkepalaBjadwalBexpiredBdibagianBdesainBbwtBbrangnyaBblackBbhanBbarangnyBveryBtersebutBspesifikasiBsinyalB
sebenarnyaBsabunBsabtuBpictureBpdhalBnoteBnempelBmicBmacetBlbihBkyaBkomplitBklwBjumatBiklannyaBibuBgtuBgameBgabisaBditulisBdilihatBcatBcardBbumbuBbijiBbarangxBantaraBaminBtanksBstandaBskrgBposisiBpokoBpcBokeeeBokayBnyobaBnyaaBngirimBmulaiBmemoryBmantapsBlingkarBlimaBlhBkalahBgapapaBexpBbruBberubahBbandungBantiBstatusBsilverBsantaiBqBphotoBpesanaBperutBpenuhBminBmeresponBlenturBkluBkantorB	ekspetasiBbuyerBbuluBbetulBbaunyaBworkBthksBthanxBsmgaBsizenyaBsemakinB
perhatikanBpedagangB	kapasitasB	kameranyaBdinginBdilapakBdayaBcontohBbohongBbesiBalasanBaamiinBzBwaBtripodBsampingBqualitasB	perubahanBpercumaBpercepatBpanduanBnontonBngaBmukenaBkakinyaBjuraganBfiturBdlBdiprosesBdikiraBdehhBdataBblsBblanjaBberasaBbeningBakuratBadanyaBwkwkBterkirimBterbukaBserviceBsamsungBretakBnyariBngasihBmenyalaBmanisBmakeBkresekBkayuBjilbabBjenisBjakaBhadiahBdibuatBbusaBbunyiBamanahBwowBtukBtemenBtadiBsulitBskaliBsgtuBsettingBsesuayBsdahBprofesionalB
pembayaranBnaikBmengirimB
memberikanBmaksihBlmynBlgsgBlaguBkepadaBkalinyaBjumlahBjumboBgrBerrorBdoankBdkirimBdeliveryBdaripadaBdalamnyaBcerahBbatereBapaanByoutubeBusahanyaB
terpercayaBtelingaBtambahanBsuamiBselanjutnyaBselamaBresletingnyaB
pengalamanBlampunyaBkwalitasnyaBjempolBintinyaBholderBexpedisiBefekBdirumahB	dibandingBdibalasBdibadanBboskuB
bintangnyaB	bilangnyaBbgdBbatunyaBalusBupdateBtingkatBterkesanBtemanBsafetyB
pengemasanB
panjangnyaBmusikBmenerimaB	mantaplahBmakanBloveBkencangBkalianB	gantunganBfisikBempukB
dipercepatB
dikirimkanBdibawaBchannelBcBbtwBbelajarBbalesBbakalanBbahwaBandroidBadminBtumpahBterutamaBterjaminBsisaBsejauhBsegeraBrokBpuassBntarB
mantapppppBlayananBlahhhBklauB
khasiatnyaBkereeenBkaretnyaBkacaBjebolBjahitBilangBdibeliB	datengnyaBbuahBbrownBbautBbahasaByngBxiaomiBupBudBtumbuhBtokonyaBtigaBsendalB	sementaraBselasaBregulerBpuasssB	ngecewainBmelihatBmaupunBmaroonB	mantaaaapBmanfaatBmalesBlumyanBlokasiBlaBkomenBketerimaBketatBkemudianBkemejaB	kebutuhanBkadoBharianBfileBdewasaB
bermasalahBberapaBbatrenyaBbangetttBajibBwBtintaBtidurBselesaiBrabuBpesananpesananBparaBpakenyaBpakekBpackBokkBnntiBngepasBmestiBmaklumBkoneksiBkendorBkabelnyaB	jualannyaBjarakBikutBhargalahBditestBcincinB
berantakanBahanBtukerBtsbB
transparanBtipeBthnksBtermasukBtempeB	tanggapanBsumpahBsukakBsolusiBslluB	sedangkanBregBrapatBpesennyaBnodaBmwBmungilBmukaBmerekamBmenyesalBmentangBmantabsBlengketB
komunikasiBknBkamisBgitarBerorBekonomisB	dompetnyaBdimanaBdijaminB
berbelanjaBbantuB	bagaimanaBbadaiBancurBwktuBwanitaBtermurahBstoreB	sebandingBsandalBresponyaBracikB
penggunaanBpahaBnikeBmmBmgknBmemoriBliburBkeringBkepuasanBkemarenB	kemahalanBgmnaBggBetalaseBditanyaBdikakiBdibawahBdiatasBdapetnyaBcadanganBblnjaB
berkhasiatBbatteryBbatraiBbankBaromaBapBwatchBtlgB
terimaksihBterbacaBsukaaaaBsistemBsisiBsetengahBsesuaiiBrpBrelatifBpudarBpkoknyaB
pemasanganBorangeBnggkBngecasBnambahBmintanyaBmewahBmenyenangkanBmasukinBmanjurBmaluBlensanyaBkerennnB	kenyataanB
kembalikanB
kekuranganBkeceBjossssBjatuhBhematBheBgayaBgandosBftoBditukarBdepannyaBblaBtrmBtotalBstikerBsopanBsesuiaBseleraBsdikitBraketBpoB
penjelasanBpenipuanBolahragaBnyatanyaBnaturalBnasiBmemesanBmemakaiBmantebBmantabbB
manfaatnyaBlokalBksihBklikBkagetB	informasiBijoBhijabBemgBditrimaB	dipackingBdikasiBdibalesBcmanBchatnyaBbnrBberkaliBberiBberaB
baterainyaB	bagusssssBapalahBakByhBturunBtdB	tangannyaBtadinyaBsudahlahBstelahBsoftBshopBsetiapBsengajaBsekianBsegiB	sederhanaBsecondBsallerBsakuBreturnBrekomendBradioBpresisiBpesannyaBpercayaBpenipuBpengirimB	pakingnyaB	ongkirnyaBnympeBnanyaBmouseBminyakBmencobaBlmyanBlistrikBkrimBjuaraBjanjiB	jaitannyaBhtBhnyaBhaturBgtBgoBgaraBdonkBdijawabBdalemBdahhBcustBcomplainBchanelBcdBbosssB	berpungsiBbeginiBbatangB
bagusbagusBapapunBalatnyaBalasBumumnyaBtxBtrimBtkBterpaksaBterpakaiBtambahinBsuruhBsiippBsekitarBsayBpusingBpriceB	powerbankBpeckingB
menerawangBmelaluiBmaterialBmantaappBmadeBleletBkupingBkrmBjawabanBjagaBinputBgramBgoresanBgarisBgagalBdigitalBcucokBcobainB
bungkusnyaBbonekaBbiayaBbarangyaBantarBadidasBwarpBwajahButuhBusahaB	untungnyaB	trimaksihBtopinyaBtiapBthisB
terimakasiBtempelanBtahunBsweaterBsungguhBslotBsisanyaBsignalB	sesuaikanBribuBpkB	pesanananBpelayananyaBpantasBnipuBngatungBmhBlmaBleherBlabelBlaahBkulitnyaBkopiBkomplenBkomentarBkkB
kirimannyaBkhususB
kependekanBkenyataannyaBkekBiklankanBgpsBduitBdsiniB	dipakenyaBdicasB	diberikanBdekatBcukurBcuciBcasingBbrngnyaB
berikutnyaBberikanBbawaanBbaguusBbagusssssssBanakkuByaituBvarianB
tulisannyaBtrlaluBtidkBtersediaBtengahBsusuaiBsudhBslimfitBserbaBserasaB	sendalnyaBsegarBrespontBresponeBpulpenBppBphoneBpensilBpaketanBobatB	ngirimnyaBnerawangBmotifnyaB
menentukanB	melakukanBmbaBmaskerBlobangBlaenB	kerusakanBkatunBkasihhBkarakterBkagakB
insyaallahB	indikatorBgooodBgigiBenggakBelastisBdusnyaBditampilkanBdikemasBdengnBckupBbordirBasikBampeBthankyouBterimahBterdapatBsyangBsuppoBsesusaiBsekalianB	sandalnyaBsampaiiBringkihBqcBpencetBnyataBnyambungBngomongBmutuB	mukenanyaBmodeBmobilBmoBmntaB
mantaaaaapBlohBkpnBknpaBkmrnBkerjasamanyaBkelamaanB	kebetulanBkaosnyaBkakaBjerseyBjasBinsyaBinstallBhriBhrganyaBgreyBgosendBgadaB	finishingBeuyBdireturBdiberiBdibandingkanBdibacaBdalemnyaBcolokanBcolokBbrooBbiarpunBbbBbaretBbangtBbangatBaudioBapikBaktifBadikBwindowsBtongsisBtmptBterbuatBtawarkanBtandaBsisirBsimBsiiiipBsichBsiaranBsiapaBsenarBsemangatBsehatBsebagusBsamapaiBsabarBrontokBrmhBremukBremotBpotonganBpositifBplayBpemberitahuanBpeganganBpasarBoutdoorBnyasarBmunculBmsihBmotorBmantulBmantabbbBlututBlurusBlemotBlainyaBkotaknyaB
kondisinyaBkerennnnBkelengkapanBkecualiBjaBisengBiphoneBikanBhmmmBhinggaBhdmiBhdBhappyBgelangBgagBfokusBekspedisinyaBditanganBdisayangkanBdesignBcoverBcorakBcaranyaBcamBbtBberwarnaBbassnyaBbaguussBanyaanByakinBwktBwahBterbuktiBtenanB
speakernyaBsngatBsmsBsllBslimBseusaiBsesuaiiiBsenterBsdkitBrejectBrecomenBputarBproblemBpekatBpajangBokokokBnmrBndaBmlhBmestinyaBmerknyaB
mengurangiBmakaBluckB	lengannyaBlariBkerudungnyaBkereeeenBkasianBjuliB	jilbabnyaBipuBikutanBhrusBheheheheBhebatBgradeBgrabBgoyangB
ditanggapiBdipeBdikembalikanBdidalamBdibilangBdanaBdaerahBckpBcctvBbukunyaBboxnyaBbotolnyaBbersaingBbasahBanalogBagustusByakBwalopunBwaktunyaButamaBuntkBtlpBthnxBtelponBtanamanBtampakBstrapBstlhBspecBsolnyaBsjBsepadanBsekedarBseadanyaBsampahBribuanBreviewBreponBrekamBpinggirB	penggantiBpdahalBparfumBnyesalBnulisBnilaiBmencariBmemilihBmelebihiBmasiBmantappppppBmamaBlngsungBlhoBkyakBkusutBkunciBkucingBkonfirmBkodeBkliatanB
kesempitanBkesanBkepanjanganBkebukaB
kebanyakanBkcewaB
kadaluarsaBjudulnyaBjariB	jahitanyaBharumBgedeanBgantengBdongkerBdiskusiBdiseBdipajangBdaunBdadaBceweBburuBbatikBbahannyBangkaBampuhBacaraBwkwkwkBtypeBtoscaB	terlanjurBtarBsusaiBspyBspoBsonyBsngtBsiaBsetBsesuwaiBsesuaBsehargaBsebagianBsachetBrepotBrecsellBrecBreaponBrapetBpribadiBpotongB
plastiknyaBpitaBpesananaBpdaBpahamBongkosBoceB
nyampainyaBnyahBntiBngepresBmuterBmuluBmlmBmieB	menyimpanBmenipuBmbakBmakanyaBmaenBlatihanBlapisanBkusamBkremBkerudungBkepercayaanB
kantongnyaBkainyaBjlnBindoorBhitamnyaBhighBhapeBhangatB	handphoneBgilaBgetBdterimaBdropshipBdropBditungguB
dijanjikanB
chargernyaBcbBbuyBbnerBblgBbengkokBbaranhBataupunBaromanyaBarmyBaaaaaaaaB
waterproofBwahanaBvariasiBuntBumumB	tripodnyaBtikiBterjadiBterhadapB	tempatnyaBtembusBtampilBsorryBsilahkanBsiiipppBsiiBsegerBseberapaB
sebenernyaBsanaB	sambunganBsambilBsamapiB	sablonnyaBresmiBreccomendedBpsnBproB	penjualanB
penampilanBpastinyaBoklahBnichBnegatifBmksBminumBmiBmerahnyaBmenuBmenjagaBmengenaiBmelayaniB	melainkanBmeBmalemBluBlanjutBlamaaaBlakuBkipasBkinclongBkhawatirBketemuBkelasBkakakBjozzBhedBhanBguysBgreatBgannBgaadaBdvdBduanyaBdrpdBdpetBditokoBdijahitBdibayarBdeskripBdengerinBdateBdarkBcinaBcabeBbukannyaBbrandBbordiranBblnBbigBbgtuBberkaratBbbrpBbandingB
bagussssssBbaguaBbagBatoBatBanycastBalisBadaptorByaaaaBwellBweifengBwattBumurBtukangBtuBtetanggaBterjagaBtentangBtempelBtaBsukanyaBstabilBsesakBseribuBsepBsampekBrodaBrendahBrekamanBrefundBpsenBpokoknyBpesnanBpergiBperfectB	perbedaanB
pengaturanBpengaruhB
penerimaanBolshopBokokBnyangkaBnyalainBntapsBnampakBmuantapBmntBmlahBminimBmilihBmenyesuaikanBmenjawabB	mengkilapBmalahanBmacamBloBldBlagiiBkurusBkomunikatifBkomputerBkeringatBkerahBkeluhanBkeluargaBkebacaB
kancingnyaBjamanBjaheBinternetBhurufB	headphoneBguntingBgerakBgatalBgamisBencerBdtangBdiorderBdikonfirmasiBdicuciBdichatB	desainnyaB	deakripsiBdayBcukuplahBcomentBcomBcbaBcancelBbyBburamBbuangetBbskBbookBbongkarBbolanyaBbgussBberlanggananB	berkurangBbercakBbangedBbalingBawettBatasnyaBappBanywayBaduhBwesBtutupnyaB	tombolnyaBtapB	tampilkanBsudaBsletingB	sinyalnyaBsensitifBsegalaBsedihBsbgBsamanyaBsablonanBruanganBresolusiB
recomandedBrbanBquickBpromonyaBponakanBpokoeBpisauB	pinggiranBpersenBperBpantesBpanasnyaBpackingannyaBngaruhBngacoBmuBmonitorBmicroBmesanBmerekaBmengirimkanB	menggantiB	mendekatiB
memakainyaBmasakBmantappppppppBmantafffBmancingBmakasihhBmakananBmagnetBlemesBlayarnyaBlagBkotoranBkoleksiBknapaBkliBkkkkkkkkkkkkkBkeyboardBketigaBketerlambatanB	kesehatanB	kerennnnnBkerenlahBkembungBkekinianBkaratanBkangBjarangBinstalBhologramBhmmB
headsetnyaBgerahBgemukBfsBfilmBenkBehhBefeknyaBearphoneBeaBdoubleBdisesuaikanB
dilengkapiB
dijelaskanB	dijadikanBdekripsiBdehhhBdcobaBdalemanBdagangannyaBcozBcoolBceritaBcalonBbuncitBbuktiBbrpBblueBbingitBbibirB	bayangkanBbatreiBbarengB	bangettttBbagetBarikBapiBanginB	analognyaBampunBaiBabangBzonkByahhBwrnanyaBvidioBvgaBuserBulasBtmnBterpisahB
terkelupasB	terkadangB
terimkasihB
tergantungBtelpBstuffBsippppBsimpanBsihhBsibukBsgituBsetaraBsesauiBseruB
semestinyaBsebentarBscreenBrubahBroknyaBrejekiBpuasaBproduksiBpriaBpokonaBpkaiBpiB
perjalananBpenggunaannyaB
pengepakanBpengemasannyaB
paketannyaBpakcingBoutBotomatisBorBoneBobatnyaBnyaaaBntrBnieBngisiBngejrengBnewBmurahhBmudhBmodisBmntpBmmgBminusnyaBminimalBmenjualBmendapatkanBmencantumkanBmantaabBlsgBlotionBlmBlhaBkurngBkurgBkulitasBkomedoBkgaB	kereeeeenBkerasaBkecilanBkamuBkacauBkacamatanyaBjawabnyaBinstanBhangBhakBgurihBgatauBframeBfdBemasBdugaanBdjiwaBditempatB	diragukanB	dipercayaB
dinyalakanB	dikantongBdicariBdibukalapakBdibantuBdganBcowokBcopyBcokelatBcheckBchargingBcewekBceptBborosBblazerBbirunyaBbhnnyaBberikutBberatnyaBberaniBbedanyaBbangusBbangeetBbagusnyaBalasnyaBalaBajhBactionBvietnamBumpanBtrnytaBtouchscreenBtlongBterlepasBtengkyuBtampilannyaBsusuBstyleBspidolBsoftwareBsmuanyaBslamatBsiipppBsesuauBsejakBsehBsebulanBsebesarBsblumBsampeiBsampaisampaiBrezekiBredBranselBrajaBpulangBpuassssBpsanB
presentasiBpremiumBpostingBpoloBpkonyaBpetunjuknyaBpesannBpenutupBpenggantinyaB	pelindungBpakaianBpahitBoveralBohBnyobainBntapBninjaBngangkatBndBnahBmngkinBmngecewakanBmeterBmesinnyaB	merugikanB
menggangguBmenempelB
memudahkanBmemberiBmaxBmantapppppppB
mantaaapppBmantaaaaaapB	makasihhhBlumayannBlumayaBlomBlistBlapanganBlamaaBkumisBkulotBkuasB	konsistenB	kondanganBkelihatannyaBkebawahBkeaslianBkaburBjnganBjaminBjadulBjackBikhlasBhujanBhpnyaBhomeBhidupkanBhandBgiliranBessenBdustyBdtengBdriverBdressBdpakaiBdkrimBditutupBdisuruhB
disarankanBdiliatBdigunainBdichargeBdibayangkanBdeviceBdetikBdeskripsikanBdapatnyaBdakBcoyBcostumerBconectBcommentBcitraBcaseBcanonBcahayaBbusukBbuatanBbrfungsiBbordirannyaBbolehlahBbogorBblurBbingitsBbibitBbgetB	belepotanBbekasiBbebanBbayanganBbatasBbasBbarngnyaBbaranyaB
barangnyaaBbalikinBbalasanBbaikkBbagusanBayoBaturBasalanBantennaBaknByaudahByachBwhiteBwasBwarungBvideonyaBtwBtuhanB	trimkasihBtrendyBtmbhBtitikBtikusBtidaBtfB	terpasangB
terdeteksiBtentuBteksturBtanggapiBsjaBsituBsingkatBshampoBshBsesuatuBsesuainBseretBseletingB
sedikitpunBsebagaimanaBsakingBsabunnyaBrutinBrespinBredmiBrecomBrapuhBraketnyaBrajutBragaBpunggungBpsBperiksaBpergelanganB	perbaikanB
pembelinyaBoverloadBordernyaBoooooooooooBnyhBnihhBngetBngebassBnexB	minimalisBmiloB	meragukanB
mengelupasB	mendinganBmemilikiBmembacaBmelesetB	maskernyaBmansetBmakenyaBlwtBlipstikBlevisBlemahBlazadaBlayuB
layanannyaBlangsingBlambanB	kurangnyaBkuduBktBkpdBkopinyaBkontrasBkomponenBkmarenBkluarBklipBkirimkanBkickersBkgkBketipuBkesekianBkesannyaB	kelebihanBkekurangannyaBkeburuBkapsulBkalemBkahBkabarBjenggotBjatohBjaringBjadilahBistrikuBinchBinboxBhubungiBhrsnyaBhamilBhadehBgreenBglassBfriendlyBfBexpetasiBefektifBditarikB
diiklankanBdideskripsiB	dibalikinBdengarBdeBdahhhBcsB	coklatnyaBchinaBchattB	ceritakanBccokBcapBcampurBcacatnyaBbolakBbgttBbgBbesinyaBbersihinBberpengaruhB
berlebihanBberkelasBberdebuBbelakangnyaBbedakBbatiknyaBbasketBbarangeBbantuanBbangeeetBbandBbahagiaBbaguuusBbackBautoBasyikBankBakhirBajalahBadminnyaBadekBaaBzoomByuntengByessByaaaaaBwaloBwajibBwajarlahBvapeButupBusiaBusahakanB	ulasannyaBujungnyaBujiBubahBuangnyaBtyBtxsBtutorialBttepBtrebleBtrackingBtnksBthankssB
terkendalaB	terdengarBtelapakBtekanBtabletBsyukronBsuksessBstandBsponBsoundBslipBskrngBsikatBsieBsholatBsesusiBservisBsemberBselisihBsejenisBsebabBsarungBsambungBsalutBsalemBsadarBringnyaBresfonBrencanaBremajaB
rekomendetBrecomentBreceiverBrealpictBreBrayaBprogramBpinggangnyaBpesanannB	perangkatBpengirimnyaBpengirimanxBpenerimaBpemakaiannyaBpabrikBotgBorangnyaBokkkBofBnyeBnyangkutBngBnetBnekoBnehBnaikinBmslhBmondayBmolorBmintakBmereknyaBmenjelaskanBmeiBmdahBmasangB
mantaappppBlumanyunBlogonyaBlmayanBleggingBlapisBlakiBlahhhhBlaaahBkuraBkrngB
kompetitifBkerumahB	keperluanB	kemiripanBkecewaaaBkecewaaBkcilBkausBjoystickBjossssssBjngnBjblBhehehBheelsBhargaiBhalalBhaduhBgudBgrosirBgripBgratisanBgoresBgoooodBgooddBgojekBgodBgimnaBgesitBgeserBgagangBfreshBfanBexpressB	expektasiBemngBdudukanBdtrimaBdsniBdownloadBdosBdkitBdiscountB
dirapihkanBdipilihBdipasarBdikirmB	dikiriminBdiisiBdidapatBdicobainBdibikinBdiajakBdescriptionBdeketBdasarBdarahBdannBdagangBcuekBchocoBcheapBcemilanB	cantumkanBbyrBbrgxBbocahBblankBbiarlahBbhannyaBbgtttBbesoknyaBberupaB	berlubangBberkwalitasBbergerakBberBbateryBbarusanBbaranggBbaramgBbagudBareaBanaknyaBacakBzipperBwlwBwirelessBwatBwarnanyBuvBtubuhBtrmaBtrimkshB	trimakasiB	travelingB	tingkatinBthnBthankzBthBterisiBterakhirBtargetBtanyakanBtanahB	tambahkanBtakutnyaBtajemBsukaaaaaBstikBsippppppBsikuBsemurahBsekecilB	sekalipunBseharusBsegelnyaBsbelumBsaringanBsampayB	sampaikanBruangBrepeatB	remotenyaBrekomBrefillB	recordingBrealnyaBrawitBrapihhBrajinBrainBpubgBpsnnBpsenanB	proyektorBpressBpouchBpohonBplugBplatBpkokBpinginBpickBpgnB	perhatianBpenyotB	penilaianB	pengrimanB	pengirimnB
pengerjaanB
pengelemanBpengaitB	penasaranB
pelayannyaBpckingBpasaranBpanahBpakinganBpajanganBpacarBolahBoemBnutupBnotaBnonBnomornyaBnntBngetikBngebulBngambilBnelponBnaroBnaisBmyaBmurBmoneyBmmcBmhnBmerdekaBmepetBmenulisBmenghubungiBmenambahBmemuasknB	memprosesBmaturBmatapBmarahBmaoBmantappppppppppBmantapmantapBmantabzBmantaaappppBmanatapBmallBmalasBmakshBmaknyusB	mainannyaBlineBlgsungBlembekBlembarBkuasnyaB	kualiatasB
kooperatifBkipasnyaBkhasiatBkerdusBkekuatanBkeepBkedipBkecilnyaBkalungBkacanyaBjmBjelaskanBjdnyaB	instalasiBimpoBidBiaBhnyBhihiBhehehheBharinyaBharganyBhargaaaBhahahahaBgudangBglowBggaBgetarBgaanBgaaBformatB	flashdiskBfixBfashionBfakeBempatBduluanBdpakeBdongleB
disebutkanBdiriBdirekomendasikanB	dimaklumiBdilayaniBdiketeranganB
dihasilkanBdiblsBdengerBdeehBdasterBdachBcottonB	conditionBcocoknyaBclanaBcacadBbulatBbukanyaBbgianBbetahB	beruntungB	berhubungBbatamBbarokahBbaraangBbaofengBbagussssssssssssBbaguBatwBaprilB	antenanyaBangkatB	amburadulBalBakurasiBagenByoByeByaahBxxxlBwlopunBwithBwawBuseBurusanBtumpulBtubeBtrustedBtravelBtouringBtopppBthxsBterussB	tercantumB
terbungkusBterbangBterbakarBtentunyaBtemperedBtekananBtarikBtanamBtahapBsyngBsupoB	sukaaaaaaBsuhuBstripBstretchBstickBssesuaiBspecialBspatunyaBspatuBsoryBskliBskaBsipppppBsintetisBsimetrisBsilauBsiiiiipBsensorBselllerBselangBsekenBseindahB
sedotannyaBsedotBsdktBsdgBsdcardBsanganBsampiB	sampaiiiiBsampaiiiBsameBsalingBrupiahBrupanyaB	rongsokanBriskanBrijekB
responsiveBrespectBresinyaBrenyahBrentanBregularBrapiiBpstiBprodakBprimaBprediksiBpokokxB
pokoknyaaaB	pokoknyaaBpnyBpkoBperbandinganBpenjepitBpenB
pemgirimanBpelayanB	pelajaranB	pekingnyaBpegelBpeaananBpayungBpapanBpantesanBpacingBoktBokoBokkkkBokeeeeeeeeeBoeBocBnyimpanBnyarisBnyanyaBnyaaaaBnuwunBnongolBnoiseBngetatBnegoBnegeriBnantinyaBnanggungBmulutBmubazirBmskipunBmnaBmknBmisalBminatBmgaBmetalBmenyukaiB	menghapusB	mengambilBmendemB
menanggapiBmemperhatikanB
membelinyaBmembalasBmedanBmbBmaunyaBmatteBmanyunBmantapppppppppBmantaffB	mantaapppBmampirB	maksudnyaBmakasiihBlumB
lipstiknyaBlightBlevelBlebihinBlebatBlagiiiBkukuBkonektorBkomitmenB	kombinasiBkoganBklaimBkhasB	ketimbangBketiakBketerangannyaBkesepakatanBkeselBkereennB	kepanasanB	kepalanyaBkepakaiBkempesBkemasB	kecewanyaBkeamananBkaratBkaraokeB	kalungnyaBjwbBjustruB	jumlahnyaBjosssssBjoranBjktBjaringanBjaitBipBinstantBinginanBinggrisBindahBhotB	holdernyaBhistoryBhihihiBhelmB
haturnuhunB
hargahargaBhandleBgunaBgoodjobBgoblokBgdeBgansBgandosssBgamisnyaBgambrBgambarrBfikirBfebB
expirednyaB	expectasiB
elektronikBdrumBdpkeBdpesanBdoraemonBdoB
diutamakanBditelitiB
ditanggungBdisanaBdipesenBdimengeBdilipatBdilapisiBdikrmB
dikomplainB
dikasihnyaBdijagaBdiiklanBdibersihkanBdiameterBdetectBdengabBdelayBdeganBdapatkanBdaahBcoupleBconveBcontrolBcocoklahBclaimBchinoBcashBcapekBcallBcairBcabutBburemBbuktinyaBbudgetBbsokBbozBbossssB	bonekanyaBbkinBbjuBbetBbersamaBbermainBberkatBberisikBberesB	berdagangBbelikanB	begitulahBbawahanBbautnyaB
bantuannyaBbangetttttttB	bangeetttB
bandingkanBbakarBbahanxBbaguzB
badaiflashBatmBatasanBasusBassalamualaikumBanyaBanjingBanginnyaBangetBandaiBamplopBamBalamiBakunB	aksesorisBahhBaesuaiBadeBabgBaaaBycBwlpunBwkwkwkwkBwkwkwkwBwkwkwBwedgesBwalapunBwadahBvalueButamakanBunyaBunboxingBukurBtouchBtoppBtoplahBtokBtnpaBtmpatBtlngBtitipanB	tingginyaBtindakanBthaksB	testimoniBtestiBterujiBtersegelBterobatiBternytaBterkaitBterjahitB	terhubungBterBtenangBtehBtaroBtarikanBtankBtagBtabBsyukaBsurabayaBsunsilkBsukasukaBsuatuBsuaiBstoknyaBstickerBstereoBstempelBstbBspeckBsobekanB
sletingnyaBslempangBsipppppppppBshgBseukuranBseuaiBsetingB
setidaknyaB
seterusnyaBseseuaiBsesaiBserupaBseriBsepedaBsepasangBsentuhBsemarangBseluruhB	selempangBsekilasBsekelasB	sekaligusBseimbangB	seenaknyaBsebuahBsblmnyaB	sayangkanBsayahBsayaaBsatupunBsarankanBsalerBroseB
retsletingBreselerBremotnyaBrecordB
recommededB
recomendetBreceivedBransaksiBrampingBrameBramBrajutnyaB	pulpennyaBpucatBpuasssssBprsananBprnhBprivasiBprinterBpolaB	pokonamahBpojokBpntingBpnjangBpngrimanBplingBplayananBpitanyaB
pesenannyaB	permukaanB	perempuanBperekatBpenyimpananB	penyanggaBpengunciBpengoperasianB	pengisianBpengalamanmuB	pembuatanBpekinganBpegangBpeduliBpedagangnyaBpeasananBpbBpatenBpasswordBpasssB	parfumnyaB
panduannyaBpandangBpancingBpadatBoyeBoutputBotakBosBorngBoksBokeokeBokeeeeeeBokeeeeeBokeeeeBofflineBoatB
nyampeknyaBnyalakanB	nunggunyaBnowBnomernyaBnnyaBngelupasBngefekB	ngechargeBngebasBngawurBnendangBnembakBmyBmustiBmusicBmurahhhBmousenyaBmoccaBmobileBmksiBmisalnyaB	mirroringBmintBmilikBmikaBmicnyaBmhonBmgkinBmfB
meyakinkanBmenyukainyaB	menyengatBmentokBmenjaminBmenilaiBmengutamakanB	mengikutiB	mengalamiBmendapatB
menanyakanBmenahanBmemuasBmembersihkanBmembawaBmelelehBmariBmantpBmantappppppppppppppBmangBmanBmampuBmamahBmalingBmaksaBmakaiBmaapBluarnyaBluamayanBlowbetBlngsngBliquidBlembabBlemanBlecetnyaBlecekBlcdnyaB
lamaaaaaaaB	lamaaaaaaBlamaaaaaBkupuBkrangBkosBkorsetBkorbanBkonfirBkitBketukerB
ketinggianBketinggalanBketarikB	kereeennnBkereeennBkepencetBkentalBkemungkinanB	kejujuranBkejadianB
kedengeranBkecillB	kecewaaaaBkcwaBkauBkatalogBjuniBjumBjrengB	josssssssBjngBjemputBjeansnyaBjarumB	jangkauanBjaminanBitupunBirBinnerBingetBingatBinfokanBhuhuBhuftBhiksBheheeBheeBheadBhbisBharumnyaBhargaaB	hadiahnyaBgunBgooooooooodBgoogleBgoodlahBgompalBgmanaBgitarnyaBgantungannyaBgannnnBgamingBgaesBfutsalB
fungsionalBformalBfocusBflasdealBfilterBfhotoBesokBentuBemailBekspresBekorBduniaBdoiBdoakanB
diusahakanBdiubahB
dituliskanBditujuanBdisniBdisampaikanB	dirapikanBdilepasBdilayarB	dikatakanBdiemB
dibutuhkanBdiaplikasikanB	deksripsiBcuteBcuacaB
controllerBcloneBciamikBchtBcepettBcepattBcenterB	cenderungBcareBcantuminBcambangBbunganyaBbungaBbuildB	bublewrapB	buatannyaBbrangxBbraBboroBbluetoothnyaBbisnisBbhBbgusssBbetisBbesarnyaBberulangBbersediaBbermutuBberkataBbergarisB
bergaransiBberenangBbentarBbeautyBbayakBbatBbarsngBbargBbarenganBbarannyaBbaranfB	bangeeeetBbagusssssssssssBbaekBbaarangBbaBazBawasBawamBatuhBasuransiBanggapBamiinBalhmdulillahB	alamatnyaBakibatBajjBairnyaBafBadainByaaaaaaBwusBwranglerBwinBwifinyaBwideBwhatsappBwhatBwebsiteBwarniBwahlB	volumenyaBviewButukBushBusaBunitnyaBukurnBukiranBtundaBtroubleB	trjangkauBtrimssBtrimksihBtrimksBtransBtradisionalBtooBtonerBtolakBtohBtlatBtipissBtindakBthkBthanksssBtestingBtesterBteruskanB
tersambungB
terpentingB
terkendaliB	tergolongBtergiurBterbiasaBterbaruBtentukanB	tengahnyaBtempoBteleponB
teksturnyaBteeimaBteamBtapiiBtapeBtapakBtanBtaiBtabungBstlahBstaplesBsrBsplitterBspinnerBspesialBspekerBspeedBsolidBsoaleBsndriBsmpenyaBsmpainyaBskrangBskBsitusBsisirnyaBsidoarjoBsharpBshakeB
settingnyaB	sesuaiiiiBsesuaeBsesekBseragamB	septemberBseperiB
sepatutnyaBsentosaB	sensornyaBsendBsemutnyaBsemutBselempangnyaBselebihB
sejenisnyaBsegtuBsecepatBsebaikB
seandainyaBscrBsaudaraBsangtBsangkaBsakunyaBsagatBrumayanBrokokBringkasBriceB	reviewnyaB	responsipBrespomBrespBresikoBreplikaBrenangBrempelB	rekondisiB	rejekinyaB
recomemdedB	recomededBrecomedBrealitaBreadBrcaBrapikanB	ranselnyaBquBputihnyaBputaranBpukulBpsnanBprintBprimeBppppBpowernyaB	postinganBpolBpokoknaBpngirimnBpkkBpintuBpingBpictnyaBpesannanBperubahannyaBperformanceBperformaB
perekatnyaB	perbanyakBpentilB	pengirmanBpenggantianBpengerjaannyaB
pengerimanB
pengecekanBpengantaranBpendengaranBpembuatannyaB	pembersihBpemasangannyaBpemakaiBpedasBpastikanBparahhBpakuBpaksaBpairingBpackibgBpackagingnyaB
orderannyaBopsiBolB
okeeeeeeeeB	nyenenginBnumbuhBntahBnotifBnightBnhBngeriBngedropBngarepB	nempelnyaBneBndkBnaruhBnagusBmuraaahBmundurBmumerBmultiBmskBmodelnyBmodalBmngknBmnBmkinBmerubahBmerakyatBmenyerapBmenyedotB	mengingatBmenghilangkanBmengeBmengapaB
mengangkatBmengakuiB	mendukungB	mendengarBmenangBmenB	memorinyaB	membangunB	memasukanBmemasangBmayanlahBmaximalB
maturnuwunBmatahariB	masangnyaBmaretB	manualnyaBmantalB
mantabbbbbB	mantabbbbBmantaaaaaaapB	mansetnyaBmandiBmaklumiBmakasiiiBmakainyaBlumayannnnnBlumanyanBlowbattBlobetBlipatanBlinkBlgsBlangananBlambangBlamaaaaBlakbanBlahhhhhBkutuBkurmaBkuncinyaBkumplitBkuliahB	kualitasxBktnyaBkriteriaBkosmetikBkonfirmasinyaBkmrinBkmnaBkmnBkmBklamaanBkisaranBkhakiBketutupBketikBketahanannyaBketB	kesulitanB	kerjasamaBkerjaanB
kerennnnnnBkelonggaranBkeliatannyaB
kekuninganB	kehabisanB	kedepanyaB
kedatanganBkebayaBkebalikBkebaikanBkebagianBkclBkawatBkasihanBkarungBkaruanBkapBkamarBkalengBkalanganBkadarnyaBjtBjozzzBjoggingBjlekBjatiBjasnyaBjaitanyaBjadikanBinfraredB
informatifBimporBikutinBikutiBikatBhrgnyaBhologramnyaBhoaxBhidungBhiburanBhewanB
heheheheheBhehehehBhargaxBhapusBhalussBhadeBhaBguuuudBgulaBgpplahBgooooodBgoooddBgokilBgmbarnyaBglossyBgileBgepengBgedeinBgdaBgataunyaBgasesuaiB	gapapalahBgannnBgambatBgambargambarB	gagangnyaBfromB	fleksibelBflasBfisiknyaBfahamBextraB	excellentBerBenteBendBenBelegantBeleBekpetasiBeeeeBeasyBdslrBdroneBdriveBdptnyaBdominanBdobelBdlamBditipuB	ditelingaBditambahkanB	disettingBdisambungkanBdipromosikanBdipkeBdipkaiBdipakekB	dimatikanBdikulitBdigambarnyaBdigambarkanBdideskripsikanB
didalamnyaB	dibiarkanBdibentukBdiaturBdiantarBdiambilBdentBdengaBdemiB	deliveredBdegBdamBdalamanBczBcurigaBcupBcreamnyaB	contohnyaBconekBcoffeeBclipB	cincinnyaBchickenBchekBchatingBcetakanBceparBcemprengBcatnyaBcasualBcargoBcargerB	cameranyaBcableBbwatBbutaBburungBburuanB	bungkusanBbuknBbrushBbrgnyBbotakBbosssssBborongBblututB	bisnisnyaBbioBbintikBbiarinBbhanyaBbgniBbgmnBbgituBbetterBbesaranBbersegelB	berminyakBberlakuB	berjualanBberhatiBbergetarBberbelitB	berbahayaBbenerinBbeltBbayiBbawelBbarcodeBbarangyBbarangnyabarangnyaBbarangnyaaaBbaragBbapakBbantingB
bangetttttBbangetsBbaliBbahnnyaBbahnBbahayaBbagussssssssssBbagusssssssssBbagussssssssBbacanBbabyBawetttBavBatsBatikBaplikasinyaBanakuBamazingB	alumuniumBallohBalhamdulillaahBalamtBaktualB	aktivitasBajaajaBajaaBafterBaccBableByyByukByoiB
yasudahlahByamanByahudByahhhBxlnyaBwudhuBwtBwrinkleBworkingBwolfisBwisBwilayahBwaterBwarnahBwantiBwalBwakaiBwaeBvoltBvirusBvalidBvacumBuploadBunBukuranxBudahlahBucapkanBtytBtugasBtrmkasihBtrlluB
travellingBtptBtoskaBtololBtmkshBtmbahBtkshBtitisBtitaniumBtikBtidaknyaBthnkBthdBthanBternyaBterimakasihhhB	terhitungBterburuBterbayarBterbalikBterawangBtensiB
telinganyaBtelfonBteledorBteleBtasikBtankyouBtanganyaBtalkBtahiBsygnyaBsxBswitchBsukaaaaaaaaB
sukaaaaaaaBsudutBsudajBstrikeB
strawberryB	statusnyaBstarsB	stainlessBssBsregBsquishyBsptuBsptnyaBspotBspandexBsoulBsombongBsolarBsodaraBsoakBsmptBsmpiBsmoothBsmapaiBsmakinBslmtBslhBskrBskinBskarangBsistaBsirsakB
sippppppppB	sipppppppBsiplahBsiphBsilikonBsiippppBshoesBshippingBsetrikaBsetipisBsesyaiBsesuaoBsesuaisesuaiBsesuaiiiiiiBserumBseriusBseratBsepatB	senternyaBsenamBsemoggaBsemigaBsembuhBselfiBsekaliiiBsekaliiBsejajarBseharianBseepBseduaiBsedotanBsedapBsebukalapakBsebelBsebanyakBscheduleBsblahBsapiBsandiskBsampleBsampeeeBsampainyBsamBsaklarBsaeBrwBrumitBrsponBroughBrollBrodanyaB	rezekinyaBrexusBreqBrentangBrendaBregoBredupBrecommedBrecomendeedBrecomendeddB	recomendeBreceivernyaB	reccomendBreaksiBrctiBratuBratingBrapihkanBrangeB
rajutannyaBrajutanBraB
putarannyaBpuringBpuaslahBpstBpotBposturBpompaBpointBpngrmnBpleaseBplangganBpkenyaBpkekBpinggulBphpBpetBpesenyaBpesaanBpermataBperasaanBpenjelasannyaB	penghapusB	pengennyaBpengelemannyaB	pengantarBpencahayaanB	penawaranBpemesanBpembungkusnyaB
pembungkusBpembelajaranBpelapisBpelanggannyaBpelanBpejualBpedesBpearlyBpayahhhBpasminaBparagonBpandaBpackingxBpackageBownerBototBoraBonsBolesBokkkkkkBogahBnyselBnyimpenBnyengatBnyedotBnyalahinBnyalahBntapzB
notifikasiBnolBningBnikonBnihilBniatnyaBnglupasBnggBngepressBnerimaBnemuBnaviBnanBnakBnaikanBmutunyaBmutarBmusimBmuiBmudikBmuchBmuantabBmrhBmotongBmonopodBmodernBmodelxBmodBmntapppBmntappBmngguBmmmBmissBmingguanBmikirBmienyaBmgBmerusakB	merupakanBmerosotBmerekomendasikanBmerekatBmenyesatkanBmenutupBmenumbuhkanBmenitanB	meningkatB
menghargaiBmengeluarkanBmengecilB	mengantarBmenampilkanBmenaikanBmempermudahBmemenuhiBmementingkanBmembutuhkanB
membungkusBmembukaBmembingungkanB	melencengB	melembungBmediumBmediaBmdBmbahBmaterialnyaBmaslahB
masalahnyaBmapBmanteppBmantavBmantappppppppppppBmantanB	mantablahB	mantaaappBmanikBmanggaBmalaBmakasiiBmahhBlusinBlumynBlucuuuBluasB	lotionnyaBlookingBlombaBlmbatBlipatBlipBleviBlemparBlemakBlegendBlebihkanBlebhBlebelBlebarnyaB	laptopnyaBlapBlanjayBlangkaBlamanyaBlalatBlakukanBlagihBkyaknyaBkunjungBkuliatasBkueB	kuantitasBkrenaBkpanBkoreksiBkorekBkoranBkolomBklasikBkirmBkingBkinerjaBkgakBkeypadBketidakBketersediaanB	ketentuanB
ketelitianBketahuanB	ketahananBkesukaanBkesiniB
kesesuaianBkerjanyaBkereennnBkereeeennnnB
kereeeennnBkereeeeeeenBkereB	kerapihanBkeongB
kenyamananB	kemerahanB	kemejanyaBkemauanBkemariBkelupaanBkeduanyaBkedetekBkecilinBkeatasBkategoriBkasihhhBkaroBkapalBkaluBkacangBjustBjugaaBjualnyaBjosssssssssssBjoobBjogingBjlasBjkBjiwaaaaaBjepangBjedaBjatuhnyaBjanggutBjamurBjajanBitungBitemnyaBiosBintiBinshaBimitasiB	imbangkanBidupBhurufnyaBhtmBhsBhmmmmBhistoriBhijabnyaBhhhBhgBherbalBheranBhemBhelloB	heheheheeBheeeBhedsetBhbsBharamB	handsfreeBhairB
hahahahahaBguudBgontaBgasBgantianBgannnnnB
gandosssssBgadgetBftBfontBfollowBfmBfiBfbBexpireB
everythingBetikaB	espektasiBesBentarBenakanBembunB	ekpektasiBekpedisiBdugaBdtgnyaBdropshipperB
dropshiperBdqnBdpatBdownBdlmnyaBdkrmBdjBdiwarnaBdiukurBdiujiBditujuB
ditransferB
ditentukanB
ditambahinBdistroB	diskonnyaBdisebutB	disamakanBdirmhBdirakitBdiputarBdipotoB	dipostingBdipikirB	diperiksaBdipergunakanBdipengirimanBdipencetBdiminumBdimensiBdimauB	dimasukinBdimakanB	dilakukanBdikrimBdikirainB	dikecilinBdikarenakanBdikantorBdijalanB	dihubungiBdihatiB	dichargerBdicantumkanBdicabutB	dibesarinB	diatasnyaBdiamBdiakaliBdgunakanBdfotoBdetilBdesuaiBdepokBdekBdealnyaBdatenyaBdatanyaBdasarnyaB
dalemannyaBdabBcuttingBcurangBcumnBcuekinBcubeBcrpatBcremBconditionerB
compatibleBcompactBcolorBcolokinBclickBclearBcleanBciputnyaBciputB
channelnyaBceperB
cepatcepatBcepanBcasanBcarBcapatBcampuranBcairanBcabangBbyarBbwahnyaBbusurBbusanyaBbulukB
bubblewrapBbuBbsrangBbsrBbsgusBbrowBbrosBbrooooBbroooB	brantakanBbpomBbosqBboskuhBborBboongB	bongkahanBbomberBbneranBblzBblhBblepotanBbleberBblasBbgtttttBbeseB
bervariasiB	bersihkanBberkesanBberkedipB	berkasiatBberhentiB
berceceranB	berbisnisB	berbentukBberbauBberbahanBberbagaiBberadaBbeludruB
belanjanyaBbegoBbebasBbearingBbatreyB	baterenyaBbataBbasnyaBbaranngB
baranganyaB
barakallahBbarBbantalBbangsatBbangetbangetBbandanaBbandaBbanBbaguuuusB
baguuussssB&bagussssssssssssssssssssssssssssssssssBbagusinBbadannyaBayamBawettttBawetawetBaturanBaseliBapekBanterBampirBallsizeBalhmdllhBaktifkanBakiBakalBajiibBajaaaaaBagBadvanB
adaptornyaBabunyaBabisssBabissBzamanBzamByonexBykkByesssB	yaudahlahByasudahBxiomiBxaBwrapingBworldBworksBweBwashBwarnaxBwangiiiBwangiiBwadahnyaBvrnyaBvisionBversiBvegasusBvansBusbnyaBusBurgentBunyuBunkBunileverBunderBultraBukrnBugB	ucapannyaBucBtunikBtumbenBtuliskanBtudakBtuchBtrmsBtrmksBtripotBtrimahBtriggerBtrfBtpiiBtopppppBtoppppBtopgojekB
tongsisnyaBtongBtokohBtnpBtlhBtkutBtiupBtisuBtipsBtimbulB	timbanganBtimbangBtidalBtiangBtiBthanksssssssB
thanksssssB	thankssssBthankaBteryataBterusssB
terkoneksiBterkenaBterimakasihhhhBterimakasihhB
terimakashBtergoresB	terbilangBterbantuBterawatBterapiB	terangkatBtepiB	tentenganBtembokBtembakaunyaBtembakauBtelorBteknisiBtekBtegasBtebelanBteaBtasxBtapuBtappiBtangkapB
tanamannyaBtampaBtamaBtaffwareBsyukurBsyaratBswsuaiBsuwunBsuksesssBsuaranyBsuadahBstrapnyaBstopBstemBstelanBstasiunBstarB
standarlahBstabiloBstBsrsuaiBsptiBspesifikasinyaBspeknyaBsoriB	solusinyaBsokBsndiriBsmpekBsmpatBslmatBslahBsktrBsklBskitarBsiyBsisinyaBsipokeyBsinkronBsingletBsingBsinarB
siiiiippppBsiiihB
signifikanBsigapBshopeeBshinggaBshareBshampooBsetengahnyaBsetelBsetauBsetalahBsesuaaiB
servicenyaBseribuanBseremBserebuBseratusBsepuluhBseppB
sepenuhnyaBseolahBsenenBsemprotBsemogaaBsembaranganBsemaunyaB	semalemanBsemalamBsemacamBselotipBsellersBselfieBseletingnyaBselaB	sejahteraB	segalanyaBsedahBsebijiBsdaBsctvBschoolBscanBsbnrnyaBsatuanBsataBsasaranBsapaiBsangetB	sangatlahBsangarBsangBsampoB
sampingnyaBsampaoBsampaiiiiiiiB
sampaiiiiiBsampaBsampBsamaaBsalonBsaiaBsahabatBsaBrumhBrubberBrouterBrokoBrobbyBrmahBrijectBrevisiBrestaBresselerBrespotBresistBresepBrekomendasikanB
registrasiBrecselBrecomandBreallyB	reaksinyaBrawanBratusanBrapiiiBrantaiBragukanBqweBquranBputerBpurpleBpungsiBpullBpulauBpuassssssssB	puassssssBpuaaaasBptgB	projectorBprofilBproduknyBprnahBpresBpradaBpouchnyaBpotongannyaBpostB	posisinyaBpolarBpolanyaBpnjualBpnjngBpngenBplgBplayerBpkknyaBpkingBpiyeBpixelBpisangBpintarBpinBpikiranBpijatBpeyokBpestaBpesankanBpesanansesuaiBpesanamBpersegiBpermenBperlengkapanBperkembanganBperihB
pergantianB	percobaanB	peralatanBperakBpenyetBpenyelesaiannyaBpenyakitB	penulisanB	pensilnyaBpenjualannyaBpenjahitBpeningkatanBpengoperasiannyaBpengokBpengirimannyBpengirimanaB	pengirimaBpenggunaBpengembalianBpengaturBpengamanB
pengaitnyaB	pendeknyaBpenangkapanB
penampakanBpemilikB	pemilihanBpelitB
pelayananxB
pelangsingBpekaBpeananBpcbBpatahanBpatBpastelBpasirBpashminaBpasanganBpasananBparisBparasutBpantsBpantaiBpalaBpaginyaBpadangBpackngBpackinyaB	packingnyB
operasikanBonkirBomongBoktoberBokokokokBokkkkkkkkkkkkBokkkkkkkkkkkBokkkkkkkB$okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeBokeeeeeeeeeeBoilBoffnyaBofficeBobrasBobengBnyusulBnyesellBnyerapBnyatuB
nyambunginBnutupinBnumbuhinBnovBnnBnitipBnilainyaBnikmatBngisinyaBnginputBngejualB	ngecasnyaBngasalBngapahBngantorBnganBngakuBnextnyaBnerBnenekBnaturgoBnasibBnasgorBnarangBnapaBmuternyaBmurniBmurajBmurahhhhBmulusssBmukenahBmujarabBmuantepBmualBmsukBmsalahBmsaBmrekBmoreBmodusBmodemBmntabBmnitBmmngBmmmmB	minyaknyaBminumanBmiminBmidiBmidB
microphoneBmessageBmesennyaB
menurunkanBmenguntungkanB	mengkilatBmenggunakannyaBmengerasB
mengatakanBmenengahBmenelponBmendadakBmencetB	mempunyaiB	mempesonaBmempercepatBmemperbaikiBmempengaruhiBmempeBmemintaB
membohongiB
membedakanB
memastikanBmemakanBmelekatBmejengBmatiinBmatappBmatanyaB
matantaaapBmarunBmaronB	manteplahBmantapzB mantapppppppppppppppppppppppppppBmantaoBmantaaabbbbBmantaaabBmantaaaappppBmantaaaapppBmantaaaaappBmantaaaaaaaapBmantaaaaaaaaapBmamakuBmaksudBmaknyosB
makasihhhhBmakashBmahalanBmaduBmaantapBlunakBlumpurBlumayaanBlukaBluangBlongBlolBlogitechBlistnyaBlisBlipstickBletoyBlensBlenovoBlekukBlekasBlehernyaB
leggingnyaBlbsBlayaniBlayaknyaBlaporBlansungBlanggengBlanBlamBlagunyaBlachBlaaaBkyknyaBkyanyaBkwlitasBkuranginB	kuningnyaBkulaitasBkuhB
kualitasnyBktikaBkritikBkrimnyaBkrennBkreatifBkrBkqBkotosBkorupBkoreanBkoreaBkontrolBkontakBkonsumsiBkomplinBkompaBkomennyaBkoinBkndalaBkmrenBkmbaliBkittyB	kirimanyaB
kinerjanyaBkickerBkicikB	khususnyaB
keuntunganBketrimaBketekB	ketebalanBketauanB	ketagihanBkesetB
keseharianBkesanaBkesalBkerutBkerennnnnnnnBkerennnnnnnB
kerenkerenB
kereeennnnB
kereeeeeenB	keponakanBkepoB	kepelapakBkenyataanyaBkendurB	kemasanyaB	kemampuanB
kekecewaanBkejutanBkejahitBkegunaanBkegiatanBkedalamBkecwaBkecubungB
kecoklatanBkecewaaaaaaaB
kecewaaaaaBkebingunganB
kebangetanBkebakarB	kdepannyaBkawanBkatakanB	kasiatnyaBkashBkasetBkarenBkanvasB	kantornyaBkalepBkainxBkainnyBkadaluarsanyaBkaaBjutaanBjumbonyaBjugakBjugBjozzzzBjoosssBjoossBjoosBjogjaBjoggerBjlsBjiwaaaBjerukBjerawatBjepitBjeliBjebakanBjeB
jazakallahBjawaBjarongBjarinyaBjangnBjangkaBjajalBjahitnyaBjagungBjacknyaBitulahBiritasiBiritBipadBindosiarBindikatornyaBincludeBincBilmuBikozaiBikatanBijinBhslnyaBhoodieBhoodBhitBhhBhehheeB	hedsetnyaBhaveBhasilnyB	harganyaaB	hargannyaB
hargaaaaaaBhardiskBhardBhadeuhBgymBgunungBgrendelBgotBgorilaBgoproB
goooooooodBgoooooodBgooodddBgoodsBgooddddBgoodddBgombrangBgoksBgojeknyaBgniBglobalBginianBgenB	gelembungB	gelangnyaBgelBgegaraBgeBgaulB
garansinyaBgantungBgantinyaBganbarBgamepadBgambaranBgamauBgamabrBgamabarBgalaxyBgalauBgacorBgabusBfutuBfuringBfotofotoBfotocopyBflashdealnyaBfiturnyaBfirmwareBfineB
fastresponBfaktorBfaktaBfairBeyeBexpedisinyaBetsBesuaiBenergenBenegBenaknyaBempangBembossBembelBemankBemakBekpedisinyaBeeeBearBdwehBdvdnyaBduitnyaBduhhBduBdtengnyaBdripadaBdpnBdontBdokterBdobleBdmnBdktBdkshBdjualBditukerB	ditemukanBditempelBditekanBdiskripsinyaBdiskBdiscBdisangkaB	disampingB	direquestB	dirasakanBdiprosesnyaBdipotongBdipindahBdipaksaB
dipaketkanBdipakeinBdioperasikanB	dinyalainBdinnerBdiniB
dimasukkanBdimasakBdilakbanBdikshBdikirinB
dikecilkanB	dijelasinBdijaitBdihpB
dihiraukanBdihargaBdigubrisB
digitalnyaBdietBdieditBdidengarBdicuekinB	dibongkarBdibibirBdibeliinB
dibawahnyaBdiawalBdiadakanBdheBdetolB	designnyaB	descripsiBdesBdengamBdcBdatngBdatangxB	dasternyaBdananyaBdailyBdaganganBdaftarBcuyBcumBcucoBcrepeBcrackBcowoBcovernyaBcounterBcosBcoretanBcopotanB	controlerB
connectionBconfirmB
colokannyaBcoilBcognosBcintaB
chromecastBchokiBcherryBchaBcetarBcepattttBcepatttBcepatnyaBcepaatBceoatBcellindoBcekungBcebanBccBcatokBcaptionBcanggihBcamnyaBcakeppBcaBbyeBbutuhkanBbusetBbunyinyaBbumbunyaBbulunyaBbuktikanBbukalapaknyaBbuffBbudBbubukBbrowsingBbrgyBbrewokBbrayBbravoBbrapaBbosterBbosbosBbosanBbooknyaBboleBbokongBbodynyaBbntukBbntangBbngtttBbnarBbluetothBbludruBblokirBblkngBbkanBbjcB	bismillahBbisingBbiniBbingittBbinggungBbiasaaaaBbiaaBbgttttBbfBbeudBbetaBberusahaBberpikirBbermotifBberkerjaBberkeringatBberkatiBberhologramBberbuluBberbekasB	berangkatB	berakibatBbenturanBbengkakBbenderaBbendaBbelahBbegBbefungsiBbedainBbdaB
bawahannyaB	bawaannyaBbatteraiB	batrainyaB	batangnyaBbatalkanBbasiBbarunyaB	barsngnyaB
barangnnyaBbarangggB	barangbyaBbaranBbangeettB	balotelliBbaloteliBbajunyBbaikkkBbahaneBbagusssssssssssssBbaeangBbaeBbableBayaBawatBauxBaudionyaBaudahBasuBasnyaBasinBasaBappleBaplikasikanBapanyaBantingBanaBamienBametBambahB
alternatifBalkoholBalhamdullilahBalakadarnyaBakulakuB	aktifitasBakirnyaB	akibatnyaBakalinBajukanBajjaBajibbbbBajhaBajeBajakBaeBadapterBactBacBabissssBzaByvByuntencBytBysByoutuberByourByneByeyeyeyByayaByaudhByantiByanhByamahaByagByaaaaaaaaaaBxpressBxperiaBxdBwujudBwrnanyasesuaiBwoofBwongBwokeBwlwpunBwlpnBwloBwlaupunB
wkwkwkwkwkB
windowsnyaBwillBwhyBwhichBwehBwebBwbBwarpnyaBwarnBwalletBwalawBwaitingBvoucherBvlogBvivoBvisualB
viewfinderBviBvariatifBvapenyaBvanillaButBusefullBusahanyB	urusannyaBultahBukurannyBukurannBukranBukrBukmBukaranBuhBugaBtvriBtutorialnyaBtutorBturuninBturkisBtuningBtumitnyaBtulisnyaBtulangBtukuBtuanyaBtsumBtsBtryBtrusssBtrussBtruckBtrnyaBtrmksihB	trmakasihBtripodnyBtrimakasihhBtreableBtrbaikBtransparantBtransitBtransaksinyaBtrackerBtrBtqtqBtpisBtotebagBtotallyBtosBtoppppppBtoopBtoiletBtnyaBtnxBtllBtlahBtkpBtjuanBtitipBtipissssBtipenyaBtidskBtidaakBtibanyaBtiadaBthumbsBthqBthpBthingsBthatBthansB
thankyouuuBthailandBtgBtetimaBtetewBtetesBtestedBterussssBterulangBterserahBtersembunyiBterputusB	terpotongBtermaBterlipatBterkirimnyaBterjualB	terimanyaB	terimakshBteriBterdugaBterciumBtercecerB
terbaiknyaB
terbaiklahB	terangnyaBtepungBtepokBtepatnyaBtenyataBtengkyuuBtengkiyuBtenagaBtemplateBteksBtekoBteknisBtekenBteimaBteganganBtegaBtdurBtdrBtdinyaBtcBtapinyaBtangkaiBtanggapannyaB	tangerangBtangBtandanyaBtahunanB	tabletnyaB	syukaaaaaBsystemBsyarBsyalBsyaaBswmogaBswiterB
sweaternyaBsusunyaBsusksesBsurpriseBsuratBsurantapBsupermanBsunfreeBsulawesiBsukuBsuksBsukkaBsukaiBsukaaaaaaaaaaaBsuedeBsueBsudanBstylishB	streamingBstockingBstiknyaB	stikernyaBstepBstdBstandbyBstaminaBsssuaiB	sssssssssBspringBspiralB	spinernyaBspinBspiB	spekernyaBspareBspandekBsoundnyaBsorenyaBsopipipipipipipippipipiBsopBsongBsoloBsoldBsoftlensBsoalxBsniBsndalnyaBsnBsmpetB
slowresponBslotnyaBslmBslimmingBslametBslamaBskippingBskalianBsjsB	sistemnyaBsipsBsippppppppppBsipplahBsinBsimcardBsiliconBsikitBsikB	siiipppppBsiiippppBsiiippBsiiiiiipBsiiiiiiiiipBsiiiiiiiiiipBsiiiBsihhhB	siarannyaBsiapppBsiangnyaBshutterBshoppingBshoBshiBshakerBshadowBsgniBsginiB	setingnyaBsetebalBsetandarBsetahunBsesuuaaiiiiBsesungguhnyaBsesuiaiBsesudahBsesuaipesananB	sesuaikahBserebuanBserbukBsepeleBsepatuxBsepatiBseparoB	sepanjangBsepaketBsensasiBsengBsendokBsenarnyaBsenapanB	senangnyaBsemugaBsemuaaaBsemuBsemgaBsemataBsemantapB
semacamnyaBselututBselluBsellersellerB	sellerrrrBselimutBselebarBselatanBselBsekliBsekiloBsekerenBsekaluBsekaliiiiiiB
sekaliiiiiBsejamBsegituuB	segitumahBsegedeBseesuaiBseenggakBsedengBsedekahBsecerahB
secepatnyaB
sebaliknyaBsealBscaraBscBsblhBsbelahBsbbBsayurBsaysBsayngBsayankBsanggatBsangadB	sampelnyaBsampeeBsampaiiiiiiiiBsampaiiiiiiBsamgatBsalinanBsalenyaBsalahnyaBsakuraBsadapBsachetanBsablonannyaBrusaknyaBrupoBrunningBrumahanBrugikanBrsBrrsponBrodBrincianBrespownBresponxB
responshipBresponnyBrespoBrespekBresetBrepoB
replikanyaBrenggangBrendamBrencengBrembesBrelevanBrekeningBrekatB
rekamannyaBrejeckB	referensiBreebokBredyBrecommandedB	recomenedBrecomendasiBrecomdedBrebutanBrebusBrebuBrbuanBrateBrapirapiBrapiinB	rambutnyaBralatBrakyatBrainbowBrahasiaBradakB
qualitynyaBqtyBqoBpusatBpundakBpuaasssBpsrBprudukBproteinBpropesionalB
promosikanBprofessionalBprodukkB
prngirimanB	prmintaanBprintingBpressureBpraktekBprBpotonyaBposnyaBpopBpoolBponselBpomadeBpomBpolyflexBpollBpokonyahBpokonyBpokokyaBpngnB	playernyaBplapakBpkokeBpketBpisaunyaBpipetnyaBpipetBpinknyaBpingpongB
pinggirnyaBpindahBpilusB
pilihannyaBpikirkanBpijitB	pigmentedBpgrimanBpgirimanBpgiBpesennBpesanyaBpesannaB	pesananyaB	pesanannyB
pesanannnnBpesanaanBpesabanBpesaBpersijaB	persiapanB	perpaduanB	permintaaBpermasalahanBpermakB	permainanBperlengkapannyaB
perjuanganBperihalB
pergunakanBperbesarBpenyelesaianB
pengurimanBpenguncinyaBpengukurBpenguatBpengrimannyaBpenglihatanB
pengirimnaBpengirimannB
pengirimamB	pengingatBpenginBpengikatnyaBpenggunaanyaBpengepakannyaBpengambilanBpengaduannyaBpencilBpencetanBpenaBpemutarBpemotongBpeminatBpemesanannyaB	pembohongBpembesarBpemberatBpembeliannyaB
pemanasnyaBpelyananBpelapajBpekigB	pekerjaanBpegangannyaBpegalBpedeBpeachBpdhBpatutBpatiBpasssssBpassssBpaslahBpascaB	pasangnyaBpariasiBparahnyaBparahhhBparaahBpantauBpancenB	panasonicBpanB	palembangBpaksainBpakingannyaBpakiBpadahlBpadBpacknyaB	packingyaBpacketBoxBoutsoleBoutdorBorinyaBoriginalnyaBorenBoptionBoptimalBoppoBoperasiBopcBooooooooooooooooooooooooooookBooooBoooBonlyBoldB
okokokokokBokkkkkkkkkkkkkkkkkkkkkBokkkkkkkkkkkkkkkBokkeBokelaahB"okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeBokeeeeeeeeeeeBoglekBodBobralBnysBnympenyaBnyeselllBnyesekBnyaringBnyantaiBnyamukBnyalanyaBnumayanBnudeBnotebookBnoraBnoobB	nongkrongBnominalBnokiaBnjingBnikenyaBniiBnihhhBnianBngurasBngulasBngpressBngiraB	ngilanginBngilangBngidamBngguBnggowoBngetesBngeselinBngerekamBngerasaBngelagBngecilinBngecekB	nganterinBngantarBngajuinBngajiBngadatBngadainBneciBndiriBnbBnarikBnapasBnaonBnangkepBnangBnameBnaikkanBnahanBnaekBmutiaraBmustBmurahinBmurahhhhhhhhhhBmuraaaahBmunafieBmumpungBmuhammadBmugoBmuatnyaBmskiBmotipBmonkeyBmodulBmobilnyaBmoalBmncBmmmmmmmBmmbantuBmkasiBmkashBmixBmirrorBmiracastBmiddleBmicrosdBmickeyBmhlBmgkBmewakiliBmeringankanBmeresapB
meremehkanBmerembesB
menyerupaiBmenyedihkanBmenyeBmenyalahkanB	menurutkuBmenopangB	menjelangB
menjadikanBmeningkatkanBmenimbulkanBmengusirBmengubahBmengsongB
mengiyakanBmenghangatkanBmenghabiskanBmenggelegarB
mengetahuiBmengesankanB
mengembungB
mengembangBmengembalikanBmengecilkanBmengecewkanBmengecewaknB	mengecewaBmengecekBmengecawakanBmengantarkanBmengaktifkanBmengBmendengarkanBmencukurBmencongB
mencobanyaB	menchargeBmencerahkanBmencapaiB	menangkapB
menanggungBmemutarB
memuasakanBmempanBmembatuBmembandingkanBmemantauB	memaklumiBmemahamiBmelorotB
melindungiBmeledakBmelebarBmekarBmejaBmegaBmdrBmdhnBmdhanBmaybeBmaxsimalBmatepBmasyaBmasuknyaBmasukkanBmaskulinBmaskBmasingBmashBmasbroBmarahinBmantapzzB	mantapsssBmantapssB%mantappppppppppppppppppppppppppppppppB"mantapppppppppppppppppppppppppppppBmantapppppppppppppppppppppppB	mantaffffBmantabbbbbbBmantaaffB	mantaabbbBmantaaappppppBmantaaapppppBmantaaaapppppBmantaaaaaaaaabbbbBmantaaaaaaaaaaaaaaaaaapBmantaaaaaaaaaaaaaaaaaaaaapBmantaBmangstabBmangapBmalkistBmalaysiaB	maknyusssBmakluminBmakeupBmakekBmakasiiiihhhBmakarizoBmakaihBmainnyaBmainkanBmaininBmahirB	magnetnyaBmadunyaBmacemBlwatBlwBlurBlumerB
lumayannnnB	lumayannnBlumayanlumayanBlumayanlaahBlumayanaB	lumayaaanBluhBlucuuB	lubangnyaB	luarbiasaBluamyanBloyoBlostBlossBlompatBlogamBlnjutkanBllahBlittleBlipsB	lingkaranBlindungiBlifeBlicikBliarBliBlgyBlgsngB	leptopnyaBlemnyBlembuttBlemariBlemannyaBleluasaBlelakiBlelahBlegoBlegingBlednyaBlebayBleastBlbrBlayaninBlawanBlarasBlantaiBlanjutiBlangkahBlangitBlancipBlampungBlaminasiBlamaaaaaaaaaaaBlakhBlahlahBlagiiiiiBlacakBlaahhBlaaahhhB
laaaaaaaahBlaaBkyoceraB	kuwalitasBkutangBkursinyaBkurangiBkuplukBkumayanBkulotnyaBkukunyaBkualityBkualitiBkualiasB
kseluruhanBksalahanB
krudungnyaBkrjaBkriukBkripikBkreeenBkpakeBkoyakBkonveksiB
koneksinyaB
komplainanBkompakBkomentB	komedonyaBkodiBkocakBknowBkmudianBkmsanBkmasanBklsBklahBkirinyaB
kickersnyaBkgedeanBketahanBkesimpulannyaBkesempurnaanB
kesempatanBkeseluruhannyaB	kesabaranBkerugianBkerrenBkernaBkeripikB?kerennnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnBkerennnnnnnnnnBkerennnnnnnnnBkereeennnnnBkeramikBkeramahannyaBkerahnyaB	kepastianB
kendalanyaB	kemudahanBkemiriBkembangB	keluarnyaBkeluarinBkeliruBkelilingBkelewatBkelengkapannyaBkelasnyaBkelapakBkekantorBkejujurannyaB	kejelasanB	kehidupanBkegencetBkeesokanB	kedodoranB
kedengaranBkedapBkecoaB
kecilkecilBkeciiilB	kecerahanBkecawaBkecapBkebunBkebenturBkebeliB
kebelakangBkawatnyaBkatakBkasikBkasihkanBkargoBkapasitasnyaBkantungBkandangBkampusBkampungBkampretBkambingB
kalkulatorBkaliiBkakkBkakikuBkadaluwarsaBkabarinBkaaihBjwbnBjuventusBjumpsuitBjozBjoyBjosssssssssBjoshBjooosssBjokowiBjobbBjntnyaBjlkBjjurBjigaBjhitanBjganB	jerseynyaBjepretanBjendelaBjempolanBjelasinBjekBjedugB
jawabannyaBjaritanBjaraknyaBjantungBjanjinyaB	jangankanBjanBjamuBjakselBjadiinBiyaaBitikadB	istimewahBiringB	instalnyaB
inshaallahBinimahBingatkanBinfonyaBindonesianyaBindomieBindoBimpianBimoBimbangBikiB	ikhlaskanBikhlasinBijazahBigBidulBidamanBibukuBibadahBhuhuhuBhuhBhubBhtrBhrpanBhrapanBhoreBhoneyBhobiBhmpirBhitungBhisapBhiraukanB	hilangkanBhigienisBhidupnyaBhidupinBhiasanBhheBhehheBhbdBhasilkanBhargB
hardisknyaBharapknBhangusBhandycamBhandsocknyaBhandfreeBhandbodynyaBhammockBhaknyaBhahahBhahaaaBhadehhBhabatusaudaBhaaBguruB
guntingnyaBgundulBgroupB
grendelnyaBgregetBgpsnyaBgosokBgooooooddddBgooooodddddBgoooodddBgoofB
goodddddddBgoodddddBgolfBgoddBgntiBgituhB
getarannyaBgenerasiBgendutBgendangBgembungBgembiraBgbsBgayaanBgausahBgatelBgaringBgannnnnnnnnBgannnnnnBgangguanB
gamepadnyaBgamenyaB	gadibalesBfuserBfunctionBfretBfrenBfreezeBfragileBfpsBfoodBfoamBflekBflaseBfitriBfinishingnyaB	filternyaBfeelBfbtBfavoritBfashionableBfaktanyaBfackingBfaceBeyelinerBexternalBexpresB
experienceBexpectBexcelentBevercossBevenBeuyyyBestimasinyaBesenBernyaBerimaBeqBepsonBepposBentahlahBenakkBenaaakBemmmBemezingB	eksklusifBegakBeehhBeehBeeBeditionBeditingBeditBedBeauBearphonenyaBdvbBdurasiBdukungBdugiBdudukBdtulisBdtngnyaBdtingkatkanBdsnBdsituBdsBdrmhB	drivernyaBdressnyaBdratBdpBdosaBdonBdolBdokumenBdockBdoainBdluuBdkakiBdiupdateBdiukuranB
diturunkanB	dituruninBditumpukB
ditulisnyaBditetimaB
diteruskanBditermaB
diterawangBditelponBditegurBditaruhBdiskonanBdisimpanBdisegelB
disediakanB	discripsiBdisconBdisayaBdisaranB
disalahkanB	dirugikanB
diproduksiBdipompaB
dipinggangB	dipesananBdiperkirakanB
diperjelasB
diperbesarBdiperbanyakB	dipelapakB	dipasaranB
dipaksakanBdipackBdiolahBdinaikinBdimukaBdimohonBdimatiinB	dimasukanBdimariB	dimainkanB	dilebihinB
dilaporkanBdilapisBdilabelBdikurirB
dikuranginB	dikurangiBdiksihB
dikonsumsiBdikoneksikanBdiklaimBdikirimiBdikiriB	dikenakanBdikananBdikameraBdijudulB	dijualnyaB
dijalankanBdijajalB	diinfokanBdiikatBdihilangkanB
dihidupkanBdiharapBdihapusBdigitBdiflashBdidlmB
didengerinBdichekB
dibelakangB
dibatalkanBdibarangBdibagiBdiatasiBdianterBdiangkatBdiajukanBdftoBdeteksiBdetakB	detailnyaBdeskrpsiBdenpasarBdenimBdengkulB	dengernyaBdengenBdengannBdengBdemikianBdemganBdemenBdehhhhhBdehhhhBdegdeganBdebuBdchBdcasBdbukaBdblsBdasiB	dahhhhhhhBdahhhhhBdahhhhBdadiBcwekBcvBcutterBcutBcustumerBcustomernyaBcumaaBcukurnyaBcukupanBcucuBcsrB	crocodileBcremeBcraBcoreBcopetBconverseB
controlnyaBcontrollernyaB
confirmasiBcondongBcoffeBcnBclnaBclingBclananyaBckBcilacapB	cibaduyutBchocolatosnyaB
chocolatosBchattingBcetBcepattttttttBcepaaatBcepaBcatridgeBcashbackBcargeBcarbonBcantolanBcanelBcajonBbysaBbyasaBbwahBbuntungBbungkukBbungBbuleBbukakB	bublewarpBbubelB
bubblewarpBbtsBbthBbrudulBbrpaBbroooooBbrgyaBbrasaBbrangnyBbracketBbqrangB	bosssssssBbossssssBboB	bluethootBblinkBblingBblesBbksBbkalBbisB	bingkisanBbinBbikBbibitnyaBbhanyBbgnBbginiBbgettBbesarkanBbersuaraB	bersamaanBbersabarBberputarBberpengalamanBberminatBbermerkB
bermamfaatBberkilauBberitahuBberhariBberhargaBberhakBbergelembungBbergayaBberfunsiB	berfaedahBberdasarkanB	berbahasaBberagamBbenyekBbenjolB
benderanyaBbenciB	benangnyaBbelmBbellBbelitBbelainBbeginianBbedaknyaBbedBbeatsBbdanBbbrapaBbberapaBbawainBbatukBbatreeBbatmanBbatereiBbarnagBbaretnyaBbarberB	barangngaB
barangkaliBbarangbarangBbarabgB	banyaknyaBbanyakkBbanyakinBbantalanBbanjirBbangsaBbanggetBbanggaBbangeutBbangetzBbangettttttttttBbangetttttttttBbangettttttB	bangetlahB	bangeeettBbangeeeettttBbangeeeeetttttBbangeeeeeetBbandelBbandarBbalonBballBbalitaBbajakanBbaiknyBbaiklahBbaikbaikBbahasBbahanyBbahBbaguuuuuuuuuusBbaguuuuuuuusBbaguuuussssBbaguuussssssB	baguussssB bagussssssssssssssssssssssssssssBbagusssssssssssssssssssssssssssBbagusssssssssssssssssssBbagussssssssssssssssBbagussssssssssssssBbagatBbagaimanapunBbadgetBbaagusBazaBawtBawesomeBaweeetBatiBatawBatapBataBasumsiBastagaBasesorisBasepBasemBaselinyaBasamBarusBareBarahBarabBaquaBapesBapahB
anycastnyaBanterinBangusBangkutBanehnyaB	amplopnyaBamperBamisBaminnnnBaminnBamiiinBamanyaBalngkahBallhamdulilahBalisnyaBalhmdlhBalhamduillahBalesanBalatxB	alasannyaBalaminBaladinBakustikBakikBakhirnyBakarnyaBakaliBahananBagsBagainBadilB	adidasnyaBademmBadeemBadaaBacehBaceBabsBabizzzBaahBzzzBzoyaBzizeBziipBzaraBzamrudByyyByupByuByraBypByouuuuByoutobeByooByogyaBykByhaByesalByepatByeeByeayBychByaudhlahByareByankByampeByampaiByamgByakniByahhhhByabgByaaahByaaaaaaaaaaaaB	yaaaaaaaaBxnyaBxdvBxbBwsBwrnnyaBwrnaxBwrnanyBworkedBwoowwBwlwpnBwlauBwkwkwkkBwkwkkBwirellesBwirelessnyaBwinnerBwihhhBwiBwhippingBwelcomeBwelaB
websitenyaBwebcamBwearBwayBwaxBwaterproofnyaB	waterpassBwatchnyaBwasalamBwarnsB	warnanyaaBwarnaaBwardahBwaranB	wanginnyaB
wangiiiiiiBwanaBwalupunBwaloupunBwaletBwajahnyaBwaffleBwaenaBwaduhBwacthBwaahhBvsBvoliaBvolBvivaBvisaBvintageBvignetteBvigentBverryBvepatBvendorBveBvariantBvallenBvagusBuyyyButaraButamanyaBusulanBusefulBusedBusapBusangBurutBurangBupnyaB	updatenyaBunukBuntujBuntuB	unlimitedB	universalBunicB	underwearBundanganBundaBumpukBumbuhanBultronB	ultrafireBultimateBulirnyaBulirBuletBulasanyaBukurnyaB	ukuranyaaBukuranyBukuraneBukuraBuknyaBukhtyBukarB	ujurannyaBujuranBujeBujananBuhukBueBudahhBucapanBubinBualBuaBtwsBtwoBtwinBtuuBturutBtuntasBtumitBtulisinBtulB	tujuannyaBtuhhBtuasBttpiBttgBttapiBttapBttBtshiBtrzBtrxBtruckingBtrrimaBtrpaksaBtrpakaiBtrobleBtrnytBtrllBtrlihatBtriplekB	trimsssssBtrimaksiBtrimakshBtrimakasihhhB	trimakashBtriBtrhdpBtrendiB	treblenyaB	treatmentBtrbuatBtrayB
transistorBtransformerB
transferanB
transfaranBtranferBtrainingBtpatBtourBtouchscreennyaBtoteBtoshibaBtoptopB$toppppppppppppppppppppppppppppppppppB	topppppppBtopmarkotopBtoplaBtopbgtBtopazBtoopppBtoooppppBtoooopBtoooooppppppBtoolBtontonBtonjolanBtongkatB	tokopediaBtoiletteBtodayBtodakBtobatBtnyataBtnyBtnkBtngglBtnggalBtmptnyaBtmenBtmbhnBtmanBtmBtlponBtlpnBtkxBtkoBtissueBtiruanBtintanyaB	tingktkanBtingalBtindihBtimnasBtigerBtidaklahBtidahBthxxBthsBthnxsBthnkzBthinkpadBthingBthenksBtheaterBthaxBthanxsBthankyuBthankssssssBtgglBtggBtgalBtetpBtetimakasihBtetehBtetapkanBterumakasihBterumaB	tersumbatB	tersenyumB	tersendatBterselamatkanBtersedotB
terperinciB
terpampangBternamaBterminalB
terlupakanBterlmbatBterlluB	terlampirBterlalauB	terlalaluBterkejutBterkecohBterjunBterjaitBteringatB	terimksihB	terimasihBterimakasiiiiihBterimakasihhhhhhhhhhhhhhhhhhBterimahkasihBterimB	terhalangB	tergangguB	tergambarBterekamB	terdetectB	terdahuluB	terdaftarBtercintaBtercetakBtercepatBterbuangB	terbenturBterbelahBterbatasB
terbarunyaB	terbanyakB
terbangnyaBteraturBteratasiB	terapinyaBteranganBtepokanBtepakBtentengBtentaraBtensinyaBtenksBtengkyuuuuuuBtengikB	tenaganyaB	temulawakBtemukanB	tempatkanBtembussBtembakanBtembagaBtemankuBtelurB	telkomselBteleskopB	teleponanB
telapaknyaBtekukB	teknologiBteknikBtehnyaBtefengBtebelinBtdnyaBtbBtawwaBtataBtasyBtasteBtasnBtasbihBtaplakBtapieBtapaknyaBtapaiBtanxBtanteB
tangkainyaB
tanggalnyaB	tampaknyaBtambhBtambahiBtamanBtamahBtamBtakkanBtaikBtahBtagihanBtactikalBtacticalBtaburB	syukurnyaBsyukuriBsyukranBsyukaaaB	syintetisBsyekaliBsyangnyaB	switchnyaBsweterBsweetBsuwonBsuweBsuuntoBsuspendBsusahnyaBsureBsurantepBsuplemenBsupermarketBsunnahBsulaBsuksekBsukayaB$sukaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBsukaaaaaaaaaaaaaaaaaaaaBsukaaaaaaaaaaaaaaaaaBsukaaaaaaaaaaaaBsukaaaaaaaaaaBsukBsuipBsugarBsuekBsudutnyaBsudagBsucsesBsuciBsuccessBsuburBsuasanaBsuamikuBsuBstudioBstrukBstreetBstrechBstraightBstocknyaBstndarBstillBstiapBstarnyaBstabdarBstaBssmpaiBsskaliBsshBsqBspreiBsprayB
spinnernyaBspinerBspidolxBspicyBspeedsBspeedoBspecnyaBspecifikasiBspcBsparepaBspanBspaiBsosB	sorongnyaBsonoBsollBsolatBsolasiBsoftcaseBsoflenBsofarBsoeBsobatBsoapBsoalnyBsnowmanBsngajaBsndrBsnangBsnackBsnaBsmwnyaBsmuleBsmuanyBsmsekBsmpnyaBsmpitBsmpaixBsmngguBsmkBsmigaBsmashBsmallBsmaaBslupBslooBslmaBsllerBslingBslideBslesaiB	slebihnyaB
slanjutnyaBskripsiBskitBskinyBskinnyBskedarBskaligusBsitBsipsipB"sippppppppppppppppppppppppppppppppBsipppppppppppppppppppppppppppppBsippppppppppppppppppppppppppppBsipppppppppppppppppppBsipppppppppppppppBsippppppppppppppBsippppppppppppBsiniiBsingleBsimpulBsimpenBsilangBsilakanBsiipppppBsiiiplahB
siiiipppppBsiiiippBsiiiiipppppBsiiiiiiippppppB	siiiiiiipB
siiiiiiiipBsiiiiiiiiiiiipBsiieB	signalnyaBsiganaBsierraBsieppBsiaaapBshowBshopingBshoeBshipBshibeiBshhBshbB	sharusnyaB	shamponyaBshadeBsgtusgtuBsezuaiBsewotBsewaktuBsetupBsetujuBsettingannyaBsetinganBsetiaBsesuwayBsesuqiBsesukaBsesueBsesuanBsesuaiiiiiiiB
sesuaiiiiiBsesuaianBsesuaaaiBsessuaiB	seserahanB	seseorangBsesamaBserutnyaBseruanB
serpentineBseriesBserialBserealB	serbagunaBserasiBseramB	seragamanBseputuBseptBsepppBsepotongBsepintasB	sepinggulBsepihakBsepetiBsepersekianB	sepelekanBsepatukuBseorangB	sentuhnyaBsensitifitasB
sennheiserBsenganBsendriBsendingBsendiBsenadaBsemutanBsemulusB
semuasemuaBsemuanyB	semuannyaBsemuahB	semriwingBsemokBsemogaaaBsemntaraBsemirBsemiBsemestiBsemerbakBsemenitBsemenB	sembarangBsemacemBselowB
sellerrrrrBselleerBselleeBsellB
selisihnyaB
selesaikanBselernyaB	seleranyaBselembarBselapisB	selangnyaBselangkanganBselamtB	selamatttB	selamanyaBselakuBselagiBseksesB	sekolahanB	sekitaranBsekerdusB
sekedarnyaBsekatnyaBsekatBsekarngB	sekaliiiiBsekaleeeBsekaiBsejutaanBsejukBsejelekBsejatiBsegelanBseganBseftyBseenakBseeBsedotnyaB	sedotanyaBsediaBsedetikBsecurityBseconBsebutB	sebungkusBseblumB
sebetulnyaBsebeningBsebenarB
sebelahnyaBseauayB	searchingB	sealernyaBsealerBseakanBsdngknBsdhlahBscraBschBscabiesBsbyBsbsBsbnrBsblumnyaB	sbenernyaBsayuranBsayanyaBsayangxBsaudiBsatelitBsatangBsasuaiBsasetBsariBsarannyaBsaraninBsarangBsapaBsaosBsanksiBsanggupB	sangattttBsangatttBsangattBsangaatBsandalnyBsanagatBsamponyaBsamplingBsampkBsamperinBsampenyBsampelBsampeeeeeeeeeeeeeeeeeeeeeeeeeeBsampeeeeBsampauBsampatBsampajBsampaixBsampaeBsampaaiBsampaaaiBsamoeBsamlaiB
sambungkanBsambungannyaB	samapinyaBsamakanBsamainBsaluranBsalhBsalesBsaldoBsalaBsajasajaBsajahBsajadahBsagaBsaftyBsadisBsadeyanBsaddleBsadayanaB
sablonanyaBsabgatBsabaranBsaatnyaBsaangatBrydenBruyekBruskBrusaBruncingBrunBrumahnyaBruksakB
ruangannyaBruakBrpiBroyalBrotiBrosyBrompiBrombakBrobbalBrizomeBriwayatB
ritsletingBritBripstokBringsekBringkesBringaBrijeckBridakBrhBrewelBreturanBretaknyaBretakanBresunBrestockBrestoB
ressletingB
responsiffBresponsibleBresponsenyaBresoonBresolusinyaB	reslitingB
reseletingB
reputasimuBreputasiBreorderB	rempelnyaBremondedBremesBremekBreleaseBrelaBrelBrekommendedB	rekommendB
rekomendidBrekomendasinyaB
rekomemdedB	rekomdasiBrekBrejekyBrefinerB	recsellerBrecommBrecomennB
recomendidB
recomendesBrecomenddedBrearBrealpicBrealityBrdaBrdBrbuBraydenBrawBratusBrapiiiiiB	rapiiiihhBrapiihB
rapihrapihB	rapihhhhhBraphB	rantainyaBramnyaBramahhBramadhanBrajanganB	raincoverBraflesiaBrafiBradionyaBrabiBqwalitasnyaBqurBquiteBqualityniceBqualitasnyaBqcnyaBpxBputihhBpusatnyaBpusarBpurnaBpureBpuraBpunyaaBpumpBpuluhBpulsaBpulaaBpulBpuckingBpucetBpuazzBpuassssssssssssssssssBpuasssssssssB
puasssssssBpuaassBpuaasBpuaaasssB
puaaaassssBptoBpsdBpsannBpsangBproyektornyaB	protectorBprosedurBpropeB
promosinyaBpromokanBprogressBprodusenBproduksinyaBproductsB
productnyaB	prodaknyaB	processorBprjlnnBpriksaB
preweddingBprestasiBpresidenBpremiereBprbaikiBpqdaBpqckingBppaBpowerfulBpowerbanknyaBpowderB	potongnyaB
positifnyaBpoporBpondasiBpondBpompomBpolosanBpollllBpolisiBpoliceBpoletBpolesBpolaroidBpokoxB
pokonyamahB	pokonyaaaBpokonyaaBpokokyBpokoknyeBpokoknyapokoknyaBpokoknyaaaaaaaaaaaaaBpokoknyaaaaaB	pokokknyaBpokokeeeBpoknyaBpoinBpohonnyaBpoetraBpodBpnyaBpnsBpnpBpngrmanBpngrimannyaB	pngitimanBpngirmanBpngirimannyaBpnBpmbelianBpmbeliBpmBplynanBplusnyaBplongBplngganBplisketBpleciBpkokxBpkoknyBpknyaBpkiBpjgBpixyBpisannnBpisahBpisaaanBpipisBpionB
pinggulnyaBpinggirannyaBpindahinBpilotBpilihkanBpiknikB
pijakannyaBpihBpieceBpicturBpicoBpichBpeyotBpesokBpesenxBpesenaBpesawatBpesanannnnnnnnnnnnB	pesanannnB	pesanankuBpesananbBpesaananBperutnyaBpersetujuanBpernikBpernakBperlukanBperlahanBperkuatB
perjanjianBperintahB	perhatiinBperformanyaBperfectoBperdanaBpercisBpercaBperbedaannyaB	perbaikinBperananBpeotBpenyokkBpenyebabnyaBpenyebabBpenyampaianB
penutupnyaBpentiBpenjepitnyaBpenitiB
penhirimanBpengririmanBpengrimnB
pengorimanBpengoprasiannyaB
pengitimanB
pengirinanBpengirinB
pengiriminBpengirimianBpengirimannyeBpengikatB
penghangatB	penggemarBpengecilBpengeBpengapBpengB
peneranganB
penempatanB	pendukungB	penderitaBpendapatBpencerahannyaB
pencerahanBpenanyaB	penantianB
penangananBpenampilannyaB
penambahanB
penahannyaBpemutihBpempekB
pemebelianBpembungkusanBpembohonganBpemberitahuannyaBpembalutnyaBpembalutB	pembacaanB	pemasukanBpemancinganBpemalasBpemakainBpelyanannyaB	pelurunyaBpelembabBpelayanannyBpelapkBpelanganBpekeBpegimaneBpegasnyaBpedesnyaBpecintaBpecananBpecaBpeasnanBpeanBpdlBpcsxBpayahhhhBpayahhBpayBpavkingBpatahnyaBpastaBpastBpasswordnyaBpasssssssssBpaspasBpasirnyaBpasifBpasarkanBpasaBparfumeBparasitBparaaaahBpapalahBpapahB
pantylinerBpantulannyaBpantulanBpantulBpantBpanjangpanjangB	panjangggBpanikBpangkasB
pangirimanB	panggilanBpanelBpandanyaB
pancingnyaBpanassBpanahnyaBpamanBpalsunyaB	palayananBpalapakBpakkingBpakingyaBpakinBpaketxB	paketanyaBpakeknyaBpakeiBpakckingB
pakaiannyaBpakagingB	padamuuuuBpadamBpadahaB	packinnyaBpackinhBpackinBpackigB	pabriknyaBoweBoverheatBoveralllBoutlineBouternyaBotongBoryBorngnyaBorgnyaBorginalB	organizerBoreoBorengeB	orderankuBoplosanB
operasinyaBonokBonoBongBomeBoliBolgaBolesinBoleBokyBokrayB>okokokokokokokokokokokokokokokokokokokokokokokokokokokokokokokB(okokokokokokokokokokokokokokokokokokokokBokokokokokokB*okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkBokkkkkkkkkkkkkkkkkkkkkkkkkkkkkkBokkkkkkkkkkkkkkkkkBokkkeeeBokkeeeBokeokeokeokeokeB
okelahhhhhBokelahhBokehhBokeeyB#okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeB	okeeeeeeeBokaBojeBojBoiyaBoilnyaBoiaBoflineBofficialB	officenyaBoesananB
oengirimanBoekBoderBobsidianBnyusahinBnyuciBnyossssBnyolotBnyoiBnynyBnympkBnympekBnymanBnyetelB
nyeselllllBnyeburBnyatakanBnyaraninBnyapeB	nyantuminBnyangkutnyaB
nyanggupinB
nyampenyaaBnyampekxBnyampeeeeeeeBnyamaB
nyalainnyaBnyaaaaaaBnutBnurahBnungguinBnumpukB	numbuhnyaBnumberBnulisnyaBnuhunnBnuggetBnuBntukBnothingBnotedB	nostalgiaBnosB	normalnyaBnormallBnormaBnorakBnopeBnoisBnocomentBnmunBnmorBnmnBniniBnikahBniihBniceeeeeBniceeBnicBnhnB	nguranginB	ngumpulinBngumpulBngukirBngrimBngresponBngopiBngonoBngojekBngmbilBngitungBngiritBnginapBngikutinBngikutBngiiingBngeshakeB	ngerasainBngepekBngemilBngembangB
ngembaliinB	ngeluarinBngeliatBngehangB
ngegantungBngegameBngechatB
ngecewakanBngecesBngebutBngeblurBngebantuBngaturBngapainBnganterBnganggurBngalaminBngakunyaBngadaBngacirBngacaBngabisinBngaaBnextimeBnewsBnettoBnettBnetralBnengBnekenBnekBnavigasiBnasionalBnamumB	nampaknyaBnambahinBnakalBnagoyaBnagihBnadaBmyshopBmutarnyaBmustikaBmuslimBmusiknyaBmusikanBmurotalBmurhBmurceBmurahnyaB
murahmurahB	murahhhhhBmurahaBmuraahhBmuraahB	muraaahhhBmunginBmulussssBmulurBmultifungsiBmukanyaBmugiaBmugaBmudahnBmubajirBmuattBmuantebBmtBmstiBmslahBmskpunBmsiBmpunBmpeBmotretBmotorolaBmotivBmotipnyaBmonyetBmontirBmontaBmonsterBmonotonBmonopoliB
monopodnyaBmonoB
monitoringBmonggoBmoncongBmogaaB
modifikasiBmodifBmodenyaBmodemkuBmnurutBmntappppBmntafBmnknBmnkinBmnjadiB
mngecewaknB
mngcewakanBmnerimaBmndingBmnatapB	mmmmmmmmmBmleotBmlembungBmkshhhBmksdnyaBmknyaBmkcihBmkcBmkasiiihBmitraBmisscomBmisrouteB	mirorlessBmirisBmiriplahBmirifBmirahBmiracleBminumnyaBminionBminggiBminesBmineralBmindahinBminalBmimpiBmilkBmilikiBmikroBmgguBmeteranBmerknyB	merhatiinB	meratakanBmerataB	merasakanBmerangkainyaBmeralB
merakitnyaBmerakitBmerakamBmeracikBmeongBmenyusulBmenyulitkanBmenyegarkanBmenyediakanBmenyantumkanBmenyanggupiBmenyambungkanBmenyakinkanBmenutupiBmenusukBmenurunBmenunjukkanB
menunjukanB
menuliskanBmenukarBmenujuBmenuhinBmenonjolBmenjebakBmenjanjikanBmengusahakanBmengulasnyaBmengulasB	mengobatiBmengkomunikasikanB	mengkerutBmengirimnyaBmengiraB
menghilangB	menghiasiBmenghasilkanB
menghadapiB
menggumpalB	menggugahBmenggiurkanBmenggelembungBmenggantikanB
mengganjalBmenggambarkanB	mengenangBmengecwakanBmengcewakanB	mengatasiB
mengandungBmengakuB
mengajukanBmengadaB
meneruskanBmenerusBmenemaniB	meneleponBmenehBmenebakB
menawarkanBmenaraB
memutihkanBmemutihBmemukauB
memuaskannBmemuaskanlahBmemprovokasiBmempelajariBmemotongB	memorynyaBmemngB
memikirkanBmemfotoBmemfasilitasiB	memeriksaBmemerahBmembulatBmembuktikanB
membukanyaB
membuahkanBmembesarBmembekasBmembayarBmembatalkanBmembaikB	membahanaB	mematikanB
memasukkanBmemanfaatkanB	memalukanBmemajangBmemainkannyaBmemadaiBmeluncurBmeluluBmelukaiBmelewatiBmeleotB
melengkungBmelatihB
melahirkanBmeizuBmegecewakanBmedsosBmecingBmdelBmbuhBmbrubutBmbokBmbkBmbiraBmbanyaB	mayoritasBmayoraBmayannnnBmaxnyaBmauveB
matursuwunB	matterialBmatotBmatikanBmateriBmatchingBmatchB
masyarakatB
masyaallahBmasterB
maskaranyaB	masjidnyaB
mascaranyaBmasakanBmarosB	markotoppBmarketBmarkBmarahiBmanyapBmantullBmantrapBmanthapBmanteppppppBmantepppBmantefBmanteeppBmanteeepB	manteblahBmantavsB	mantapzzzB
mantapssssBmantappsBmantappppppppppppppppppppppppppBmantapppppppppppppppppppppppppBmantapppppppppppppppppppBmantappppppppppppppppppBmantapppppppppppppppppBmantappppppppppppppppBmantapppppppppppppB	mantapnyaB
mantapjiwaBmantapinBmantapantapBmantapaBmantaftBmantaffffffffB
mantafffffBmantabsssssssB	mantabsssBmantabssBmantabbsBmantabbbbbbbbbBmantabbbbbbbbBmantabbbbbbbBmantaapppppBmantaafBmantaabbB	mantaaapsBmantaaappppppppppppBmantaaaappppppB
mantaaaaabBmantaaaaaaappppppBmantaaaaaaapppppBmantaaaaaaappppBmantaaaaaaaapppppppppBmantaaaaaaaaaaapBmanstapBmanstabBmaniaB	mangkanyaBmandekBmanapunBmamtapBmalsBmalmBmalinauBmaksiBmaksBmaknyussBmakmurB	makasiiihBmakasieBmakannyaB
makanannyaBmakaasihBmakBmafiaBmadiunBmadepBmacroBmacbookBmabtapBmabarBmaahBmaafkanBmaacihBmaaafBlymayanBlwbihBluvBlusinanBlunasB	lumyanlahBlumbayanBlumayunBlumaynBlumayannnnnnBlumayanlahhhBlumayanlahhB	lumayanlaBlumayamB	lumayaannBlumayaaaaanBlulurBlugasBludesBlucuuuuuBlucuuuuBluberBluaamaB	luaaaaaarBlsngBlpakBlowbatBloufieBlotBlorengBlooksBlookBlokBlogitecBlodayaBlockBlnyaBlnsngBlnjutBlngsgBlngkapBlnggananBlmnyanBlmbutBlmaaBlksBliveB
listriknyaBlipetanBlingkarannyaBlinerBlincahBlimitedBlimbnyaBlimBlilitnyaBlilitBliburanBliatnyaBliangBlhoooBlezatBletakBletBlestBlerenBlengkpB
lengkapnyaBlengkapiBlenBlembaranBlemanyaBlehB	legingnyaBlegaBlebiBlebelnyaBleatherBlearningBleBlbBlazypadB	layananyaBlawBlaumayanBlasBlaporkanBlaporanBlaperB	lapaknyaaBlapakkBlantunanBlantasBlandingBlanaB	lampirkanB
laminatingB	lambatnyaB
lambangnyaBlamaaaaaaaaaaaaBlamaaaaaaaaaaBlamaaaaaaaaaBlajuBlainkaliBlahhhhhhhhhhhhhhhhhhhhhhhBlahhhhhhhhhhhhhhhhB
lahhhhhhhhBlaginBlagiiiiBlagianBlaggingBlacosteBlabelnyaBlaahhhhBlaaahhBkynyaBkxBkuyBkuterimaBkususBkursorBkurmanyaBkurisB	kupluknyaB	kupingnyaBkuotaBkunyitBkuncianBkunBkumurBkumpulBkulkasB
kulitasnyaBkulakanBkudaB	kucingnyaBkucelBkuarBkuantitiBkualtasBkualotasBkualitetBkualitanBkualitaBkuakitasBktmuBktanyaBksongBksBkrudungBkroposBkretekBkrenyekBkreenBkreeennnBkreditBkreatifitasBkoyoBkotexBkosletB	koperatifBkoperB	koordinatBkonveB
kontrolnyaBkontolBkonterBkontenB
konsumenkuBkonsletBkonsenBkonfrmBkondsiBkomporBkompetitiveB
kompensasiB
kompatibelBkomitmennyaBkolamBkohBkogBkodokBkoclakBknaB
kmungkinanBkmuBkmbliBkmarinBkmanaBklyBklowBklotakBklopunBklopBklokBklihatanBklienB
kliatannyaBklemBklarifikasiBkkkkkBkkiBkitkatBkitimanBkirmanBkiriminBkirangBkiranBkiraiinBkiplingBkinyisBkiniBkiloanBkiloBkilapBkikirBkiaraBkialitasBkiBkhoirBkhitanBkhimarBkhairanBkeyBketumpukBketujuanBketuaanBketombeBketokB	ketiganyaBketenBketempatBketawaBketapelB	ketangkepB	ketajamanBkesellB
kesayanganBkesayaBkerutanBkerusakannyaBkerupukB
kerontokanBkernBkerlipB
keripiknyaBkerinBkeretaBkerenzzzBkerenzB*kerennnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnB&kerennnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnBkerennnnnnnnnnnnnnnnnnnnnnnnnnBkerennnnnnnnnnnBkerenanBkeremBkereeennnnnnnnnnnBkereeennnnnnnBkereeeeennnnnBkereeeeennnBkereeeeeeeennnnB#kereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeenBkerebBkerapianBkeranB	keramahanBkepotongBkepingBkepakekBkenyataannyBkenyalBkentutBkentaraBkenpaB	kendaraanBkendangBkencingBkemiripannyaB
kemiringanBkemerdekaanBkembarB	kembaliinB	kemanakahBkemajuanBkelupasB
kelunturanBkeluarannyaBkelinciBkeliatBkelemahannyaB	kelemahanB	kelecilanB	kelalaianBkekuranganyaB
kekuatanyaBkekuatannyaB
kekeliruanBkekecilaaanBkekarBkejepitBkejauhanB	kehujananB	kehitamanBkehijauB	kehendakiBkehendakBkegunaannyaB	kegagalanBkeempatB
kedinginanBkedetectBkedepnBkeciumBkecilllllllBkecikBkeciiiilBkeciBkeceweB	kecewalahBkecewakBkecewBkecengBkecekikBkeceeeBkeceeBkecambahB	kebotakanBkebihB
kebersihanB
keberkahanB
keberadaanBkeasliannyaBkeangkatBkeabisanBkdepanyaBkdangBkcewaknBkbutuhanBkbsaranBkbesaranBkayunyaBkaykBkatarakBkatanyBkatampiBkatagoriBkataaBkasusBkasiiihB
kasihkasihBkasihhhhBkaryawanBkarmaBkarepBkardiganBkarawangBkapurBkapokkBkapasnyaBkantoranB	kandunganBkanakBkamsiaBkamipunBkamejaBkalimatBkalehBkalaupunB	kaitannyaBkaitanBkailBkaiaBkadihBkabehBkaaaaBjwBjugaaaBjudesBjuaraaaBjualinBjualanyaBjozzzzzzzzzzzzzzzzzzzzzzzzzzzzzBjoystikBjoykoBjowoB!josssssssssssssssssssssssssssssssBjosssssssssssssssssssBjossssssssssssssssssBjosssssssssssssssssBjossssssssssssssssBjossssssssssssBjossssssssssBjossjossBjoossssBjooosB	joooossssBjooooooosssssBjoooooooooooooooossssssssBjonBjombloBjokBjoinBjogerBjobsBjminnBjlBjiwalahBjiwaaaaBjiwaaBjiplakBjingkrakBjilbabxBjhB	jerawatanBjepretBjepitanBjenisnyaBjengkolBjengkelBjeleknyaB
jelekkkkkkBjelekkBjedagBjeandBjazBjawabkanBjauBjatimBjarumnyaB	jarongnyaBjargaBjantanBjanjikanBjangkisBjangkauBjamuranBjameBjambangB	jahitannyBjagoB	jadwalnyaBjadualBjadBjacketBjabodetabekBizinBiyahBivesBistimasiBisteriBistanaBisolasiBisinyBisaBinuBintruksinyaB
intropeksiBinternasionalBinternalB	interfaceB	intensiveBintelB	instruksiBinstalasinyaB	inshallahBinputnyaBinnernyaBinjBinilahBiniiiBinihBinginknBinggalBingetinBinfraBinfinixBindukB	indramayuBindorBindomaBindofoodBindikasiBindiaBinciBinaraBimproveBimingBimageBilanginBilanganBiketanBiketBikehB	ikatannyaBikannyaBikalBijonyaBiinBiiiiiBiiBihBidupinBidentikBicBibaratB	ibadahnyaBhyBhuuftBhumanBhumairaB	hubungkanBhubunganBhuBhttpBhtamBhsilnyaBhsdpaBhrpnBhrpBhpxBhoweverBhowB
housingnyaBhourBhotelBhormatB
horizontalBhooqBhookBhoodnyaBhohoBhncurBhnBhmprBhmmmmmmBhmmmmmBhmBhlusBhitunganBhikingBhijaunyaBhihiiBhihahBhightBhighlyB	hidungnyaBhheeBhhbBhexB	herbalnyaBhemmB
helikopterBheheheheheheheheheheBheheheheheheheheBheheheheheheBheheheeBheheeeBheehheBhedehBheadphonenyaBhdupBhddBhbjBhawaBhatikuBhatgaB	hasilnnyaBharussBharusnyBharuaBharpanBhariiiBharhaBharganyalahBharganyaharganyaBhargaaaaaaaaaaaaaaBhargaaaaaaaaaaaBhardisB	harbolnasB
harapannyaB	harapakanBharapaB	haraganyaBharagaBharBhapenyaBhapalBhanyakBhankBhangoutB	hangatnyaBhandukBhandsockBhandsetBhandmadeBhandbodyBhambarBhalamanBhairdryB	hahahhahaBhahahhaBhahBhaedsetBhaduuhBhadeuuuhBhadeuuhhBhadeehhBguuudBgusB	guritanyaBgunanyaBgunaknB	gunainnyaBgunainBgulungannyaBgulitaBgulanyaBguitarBgugelBguddBgubrisBgtuuBgthBgsmbarBgsmBgsB	grosirnyaBgripnyaBgrayBgrandBgrafiranBgppaBgplBgospolBgosokanBgorillaBgoresnyaBgoplusB%gooooooooooooooooooooooooooooooooooodB#gooooooooooooooooooooooooooooooooodB!gooooooooooooooooooooooooooooooodB goooooooooooooooooooooooooooooodB!gooooooooooooooooooooooodddddddddBgoooooooooooooodBgooooooooooodBgoooooooooodBgoooooooooddddB	gooooooodBgoooooodddddBgoooooddddddB	gooodddddBgoooddddBgooglingBgoogBgoodjobsBgoniBgnBgmbrnyaBglutaBglowingBglassesBglasBgladBgkaBgitulahBgimnBgimbalBgimanahBgilaaaBgilaaBgigitBgiginyaBgigaBgiftBghofuraaBggwpBgeterBgetaranBgesperBgesekanBgemesBgembosBgembelBgelnyaBgeliBgelengBgedekBgdBgcBgbsaBgaysBgaxBgarutBgarukBgaretBgaragaraBgappBgapernahBganyeselBgantiinB
gannnnnnnnB	gannnnnnnBganjelanBganjelBganjalannyaBgangguBganganB	gandossssBgandossBgamvarBgamuatBgambbarBgambarrrB
gambarnyaaBgamBgalakBgakbisaBgajelasBgaitersBgaharBgagahBgaesssBgaeBgadingBgabungBgabarBgabBgaaaaaaaaaanBfruitBfrozenBfrontBfotiBfokusnyaBfoBfluidBflexibleBfleshBfleeceBflatBflammingBflBfishingBfireBfingerprintBfilenyaBfiikumBfifojamBfictBfhdBfereyBfeederBfeedbacknyaBfeedBfedbackBfavoriteBfashBfaseBfantaBfalsBfallBfakingBfaBezcastBeyB
expressnyaBexpoBexpectedBexpectationBexcitedBexBevodBeuyyBessenceBesokanBeropaBerahB	equalizerBepatBeosBengselBengkolBengganyaBendolBenalBenaakBemulsionBemenBembosBemblemBelektrikB
electronikBekstraBekoBehemBeeeeeeBeeeeeBedukatifBedukasiBedisiBedannBechoBearpickBearcupBdwnganBdusboxBduplikatBdunkBdungBduhB
dudukannyaBdudeBdudahB	duaduanyaBdtgnyBdstBdryerBdrumnyaBdrpdaBdrillBdrawingBdputerBdplastikBdpanBdosnyaBdoneBdombaBdokumentasiBdoffBdocomoBdnaBdmintaBdmBdluarBdlnaBdllnyaBdliatBdlashBdkrmnyaBdiwajahBdiutakB	diupgradeBdiulasBditwrimaBditvnyaB	ditutupinB
ditungguinBditingkatinB	ditinggalB	ditimbangBditetesBditerimakanBditerimBditepatiB	ditawarinBditawariB
ditanyakanB	ditanyainB	ditangkapBditanggepinB
disurabayaBdisukaBdiskipsiBdisituB	disiniiiiBdisiniiB	diserviceBdisepatuB
disepakatiBdisenterBdisemirB
diselipkanB	diselipinBdiselesaikanB	disekitarBdiscontB
disconnectBdiscoBdisappointedB	disambungBdisaatBdirubahB	diruanganBdiruangBdiretureB
direcomendBdirandomBdipisahB	dipinggirBdipindahkanB
dipilihkanBdipihakBdipictBdiperutBdiperhatiinB
diperbagusBdipenuhiBdipegangB
dipastikanB
dipasarkanBdipasangkanBdipanjanginBdipakingB
dipakaikanB	dipakaigaBdipahamiBdiolesB
dinyatakanBdinilaiBdinerBdinasBdinaB	dimurahinBdimnaBdimanfaatkanB	dimanapunBdimaksudBdimB
dilindungiBdilemB
dilebihkanB	dilapisinB
dilapanganB
dilapaknyaBdilacakBdikunciBdikonfirmasikanBdiklikB	diklankanB	dikirimnyBdikiriminnyaBdikimB	diketahuiBdikepalaB
dikemudianBdikecewakanBdikeB	dikasinyaBdikarenaBdikardusBdikantonginB	dikampungBdikabariBdikaaihBdijadwalkanBdiinjakB
diingatkanBdiinformasikanBdiinfoBdiikutiBdihitungBdihariBdihargaiBdiharapakanB	diguntingBdigosokBdigmbrBdigmbarB	digerakanB	digenggamBdigedeinBdigbrB	digaransiBdiganjalBdigabungBdifungsikanBdiformatBdiflashdealB	differentBdietnyaBdidugaB	didisplayBdideskripsinyaB
didalemnyaBdidakB	diconfirmB	dicolokinBdicolokBdicheckBdichasB	dicatatanBdicatB
dicantuminBdicampurBdibwB
dibuktikanBdibuangB	dibterimaB	dibonusinB	dibolongiB
dibohonginB	dibohongiB	dibenerinB	dibatalinB
dibanyakinB	dibandrolBdibalutBdibalikBdibakarB	dibagikanBdiareBdiangkutBdianggapB	dialihkanBdialamatB
diaktifkanBdiakalinB	diafragmaBdiaduB	diabaikanBdgambarBdfnBdeuiBdetektifBdetekBdeskripsionBdeskripsideskripsiBdeskripsBdeskripiB	deskribsiBdeskipsiBdesignyaBdesemberBdescripBdescBdesaignB	depkripsiBdeoBdennganBdengungB	dengarkanBdengannnBdengahBdenfanBdenBdemoBdellBdeliverynyaBdeleveryBdelBdekkerBdekilBdekerBdehhhhhhhhhBdehdehBdefenderBdefectBdeehhBdeechB	dedkripsiBdedeBdebunyaBdearBddrBddanBddBdcuciBdcbBdbwaB	dbutuhkanBdbungkusBdbntuBdblgBdaunnyaB
datascriptBdatarBdatanggBdatabgBdaptBdapetinBdansBdangBdandanBdampaiBdalmBdalanBdahhhhhhBdagingB	dagangnyaBdaduBdadakanBdaanBdaahhhBdaahhBcwokB
cuttingnyaB	customersB
customerkuBcusBcurhatBcukBcuakepBcropBcrocsBcreditBcrayonBcrapBcpuBcoyyBcowBcountingBcoruptBcopyanBcopasBcontourBcontinueBconsumenBconnekB	connectorBcongBcompareBcomfoBcomenBcolourBcokiBcodeBcocokkBcocogBcocockBcobahBcobaaBcoBcntohB	clutchnyaBclnanyaBcleaningBclassicBckckckBciumBcitizenBciriBcirebonB	cintailahB	cingkrangBcingcongBcingBcindyBcincayBcikiBciamiBchitBchetnyaBchasB	chargenyaBchanellB	chamomileBchamberBchBcfdBcetakBcesB	ceritanyaBceritainBcerdasBcepotBcepetanBcepattttttttttB	cepatttttBcepagBcepafBcepaattBcepBcentiBcembungBcelnaBcelanayaBceckBceBccdBcattonB	catokannyBcatokanBcassBcasnyaB	casingnyaBcasenyaBcarpetBcariinB	cargernyaBcardnyaB	cardboardBcaramelBcaptureBcappeBcapeBcapacityBcantixBcantikkBcantiikBcangkangBcampingBcamelBcakepsBcakepppBcakB	cairannyaBcahtBcadangannyaBbyaBbwhBbwaBbuyersBbuwatBbututBbutuhnyaBbutiranBbutirBbutekBbushnellBbushingBbusanaBbusBburgerBbungkusannyaBbundarBbumilBbumiBbulumataBbuletBbulatnyaBbulatanBbulananBbulakBbulBbukamallBbukalapaBbukalaB
bukaklapakBbukaaBbujetBbuiltBbugusBbuffnyaBbudgedBbuburBbublleBbublesBbublenyaB	bubblenyaBbuataBbuangedBbuagusBbuaBbtamBbsoknyaBbsarBbrushnyaBbruleeBbrowserBbrosurB	broooooooBbrngnyB	brmanfaatB
brkualitasBbrisikBbrikutB
brightnessBbrightBbricaB	brgsesuaiBbravooBbraketBbozzzzBboxerBbosskuBboskuhhBborderBbootsBbootBboosterBboomBbonusxBbonusnyaaaaaBbonusnyB
bonusannyaB	bombernyaB	bombastisBbomBboluBboltB	bolongnyaBboleehBbodohiBbodiBboboBbntrBbntgBbngttttBbngttBbngettBbngeetBbngeeetBbngatBbmB	blututnyaBblurrBbloomBbloodBblonBblkgBblissBblikinBblgnyaBblezerBblenderBblehBbladeB	blackhawkB
blackberryBbkumBbkaBbjunyaBbjBbiyungBbiyarBbisalahBbisakahBbiotinBbioskopBbioreB
bingitzzzzB	bingittttB	bingitsssBbinggoBbikinnyaBbidangBbibirnyaBbiasanyBbiasaaaBbiasBbiangBbhwBbhsBbhnyBbhnxBbhkanBbgtttttttttttttB	bgtttttttBbgiBbezelBbetulllllllllllllBbetullBbetulanB	betternyaBbetadineBberwajibB	berukuranBberujungBberuangB	bersyukurBbersifatB
berpergianB
beroperasiBberniatBbernamaB
bermanpaatBbermanfaB	bermaksudBberlapisBberlabelBberkrgBberkerutB	berkepingBberkenanBberkelanjutanBberkecimpungB	berkameraBberjayaBberjanjiBberisiBberhijabB	berhadiahB	bergoyangBbergelombangBbergantiB	bergairahBberfungsinyaBberfungiBberetB	berdungsiBberdiriBberburukBberbunyiBberbuatB
berbandingBberattBberasBberangB	bepergianBbepBbentolBbenjolanBbenihB	bengkoangBbengBbelymBbelunBbeloknyaBbelingBbeliinBbeliiB	beleberanBbelangBbelahanBbelaBbelBbekaliBbeigeBbeeBbedaaBbecauseBbeberapakaliBbebekB
bearingnyaBbdnBbdgBbcaBbbrpaBbbmBbbbbbBbbbbBbbarangBbayangBbawangB	bawahanyaBbaudBbateriBbatchBbatalinBbastBbaruuBbarleyBbarjadBbarisBbariB	barcelonaBbarangnaBbaranganBbaragnyaBbaraBbapanyaB
banyuwangiB
banyakkkkkBbanyakkkBbanyakanBbanyaaakBbantenB	bantalnyaBbangkrutBbanggBbangettttttttttttBbangettttttttBbangeeettttBbangeeeetttBbangeeeeeeetBbangeBbangauBbangaaatBbandrolB	bandinginBbamgetBbambunyaBbambuBbambooBbalonnyaBbaliknyaBbalesnyaBbaladoBbakalnBbajuxBbajusB	bajunyaaaBbajinganBbajetBbaikkkkkBbahuBbahannyabahannyaBbahannBbahamBbahalBbaguuuuusssssB	baguuuuusB
baguuuusssB	baguuusssBbaguusssB,bagussssssssssssssssssssssssssssssssssssssssB%bagusssssssssssssssssssssssssssssssssBbagussssssssssssssssssssssssssBbagusssssssssssssssssssssBbagussssssssssssssssssssBbagusssssssssssssssB
bagusssssaBbagussbagussBbagusbagusbagusbagusbagusbagusBbagsBbagoesBbagasBbagaiBbaeeeB	baeangnyaBbadankuBbackupB	backlightBbabonBbabiBbabelBbaangBayuBayooBayatBayahBayBaxisBaxeB	awettttttBaweettBaweetBaweeeeeeeeetBawaetBauxnyaB	authenticBausBatutBatasiB	atasannyaBataoBaslinyBaslianyaBaskBasesorisnyaBasapnyaBasapBarrivedBaroganBarmBarangBaquaticB	aquascapeBappsBaplBapkhBapkBapalgiBapaaBanythingBanycasBanunyaBanuBantvBantraBantigoresnyaBantemBantapBannyaBanjurkanBanjirBangkapBanggungjawabBangBanekaBandromaxBandalkanBandalanBamsyongBampliBampatBampBamiinnBamazonBalurB	aluminiumBalquranBalphabounceBalmatB	almamaterBallahumaB	alhmdllahB
alhamdullhBaleBalbumBalarmBalamB
akurasinyaBakurBakunyaBakuiB	aktualnyaBaktivBakhiratBakhiBakarBakangBajjahBajipBajiiibbBajibbbBajahhBajaaaaaaBajaaaBairplayBaibonBahhhBahdBahaninBahahahaBahahaBagxBagustsBagannyaB
affordableBaduhhhBaduhaiBaduBadobeBadlhBadlBadjustBadhBademmmBademademBacunginBactivityBacraBabusBaboutBabizzBabcdefghijklmnopqrstuvwxyzBaadaBaaamiinBaaaaaaBaaaaaB$zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzBzzzzzzzzzzzzzzzzzzzzsssssBzzzzzzzzzzzB	zzzzzzzzzBEzzzzzoooozzzzzizizizizizixjejrjfjfkgogjdhhshdfhhdhdhdhdhdjfjfjfjfjfjfBzzzzB
zzzxxaskedBzzztttttBzznzbsbsbhjabsgssnnaBzzB zxxxxxxxzzzzzzxxxxxxxccvbbnnvcfgBzwnfomeBzusjsjdjBzosssssBzosssBzossBzongBzjiangBziseBzippppBziplockBziperBzilongBzilingoBziiiplahBziiiiiipBzhsjsjsB#zhshshshhshsshbsbsbsshsjxnxndhdhsjsBzhsBzhiyunBEzgvsbsixhsbaizusbbizuzbsuxyshixuxbsnziusbskzizgbxkcidhskaoushakzoxoxkBzgpaxBzezuaiBzesuaiBzerrrrrBzepatuBzenithaBzenfoneBzemogaBzeBzbdhdguwjsvdhsjdjdjfhBzamzamBzaitunBzaimBzahraB�yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyuuuyggvxzsfvcxjhzuxkckB%yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyByyyyyyyyyyyyyyyyyyyyyyyyyaB'yyyyyyyyyyyyyttghjkkjgfgghhhghjhjhgggggByyyyyyyyyyyyyesByyyyyyyyyyyyByyyyaaaaaaaByyxyijxdgjhduibcfuB'yyhhjjhhjjsjsjbejdjjdjsnsjskskskskskskjByyangByyaBywdhByuyuB
yuwushehhdByutubeByutubanByutubByutengByupzzByupsByuoByuntfngByummieByuiopasdfghjklzxcvbnmqweByuiopasdfghjklzxcvbnmByuiopasdfghjklzqweByuiopaByuiopByuaaaBysngByslByskinBysdByratzkxeByqByoyoyoByoyonyaByoyoByowislahB	yoweslahhByowesByoutebeByoouuuByooseeByoooosssB yoooooooooooooooooooooooooooooooByooooooooooooooooB	yooooooooByooooByongnuoByonekByomanByokByoguByogaBynhBynBymgakBylongBykyByiolllByihuuuuByidakByiByhuuuuByhcBygxgjfghibvguftBygvdiBygsobekB
ygbdikasihByeyeyeByeyeyByeyeByeyByeuhByetByesyesyesyesyesB(yessssszzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzByesssssssssssssssssssssssssByesssssssssssssssByesssssssssssssB$yessssssssssdsssssssssssssssssssssssByesssssssssByesssssB	yessfhjklB
yeselyeselByeselByernyataByerimaByellowByeeeeeeeByeeeByeayyyByeaaaaahByeaaaByeaBydmBydhBycoycoycBycouvByayayyaB#yayayayayyayayayayayayhgtggddgjjgddB"yayayayayayyayayayayayayayayayayayByayayayayayayayayyayayayaByayayayayayayayayayayyyuyByayayayayayayayayayayahaB$yayaayayyayayayayayyayayayayayayyayaByayaaaaByayByaudalahByaudaByatimByasudhByasudaByarobbalByariByapiByaowohhhByaopoByanvByantoB!yannnnnnnnnnnnnnnnnnnnnnnnnnnnnnnByangsByangkaByanggByangberbicaraByangbdiByanaByanByampenyaByamadaByalByakkByakinnnnnnnnByakbadaByakaliByaiyalaByahuuddByahudddByahoodByahoB
yahlumayanByahhhhhhhhhhhhhhhhhhhhhhhhByahhhhhhhhhhhhhhhhhhB	yahhhhhhhByahaaaaaaaaaaaaaaaByagituByabByaayaaByaangByaahhByaaapByaaakkkByaaahhByaaaahByaaaaahByaaaaaahhhhhhByaaaaaaaaaqByaaaaaaaaaaaaaaaaaaaaaaaaaByaaaaaaaaaaaaaaaaaaaaaaByaaaaaaaaaaaaaaaaaaaaaByaaaaaaaaaaaaaaaaaaByaaaaaaaaaaaaaaaaaByaaaaaaaaaaaaaaaaByaaaaaaaaaaaaaaB
yaaaaaaaaaByaaaaaaaB$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxBxxxxxxxxxxcccccccccBxxxBxxiBxxBxttBxtraBxtBxseqtuugBxpektasiBxoxoBxnyBxlnyaaBxlipatBxkdbddB0xjsiskbsvxjwoskskshxksnzhjsbszbzjzjbzhzmsbznsknwBxjddB
xixixixixiBxixixiB	xiaominyaBxiaomaiBxiamiBxhydtegwtgzgdywhB
xhshauagbaBxhBxfdbbegrwsdvrdgfbffdsfeBxehoBxanBxampeBwwowwBwwowBwwoooBwwkwkwkwkwkwkkwkwkwkBwwkwkwkBwuuuookkkeeeeeeeeeeeeeeeeeeehhBwuokeyB	wulungnyaBwulungBwujudkanBwuihBwuduBwudluBwsterB)wsksmsoskososjsjskskmwmsksksoskksmeksmsosBwsierBwscnyB
wrnanyaaaaB	wrnaanyaaBwrappB
wrapingnyaBwrapeBwrangerBwranaBwrabBwrBwqrnaBwoyBwowwBwornBworkmanshipBworkkkkkkkkkkkkkkkkkkkkBwoowBwoooowB	wooooooowBwoooooooooooowBwoooooooooiBwookeeeBwookBwoodB
wondowsnyaBwolvisBwololooooooooooooooooooBwollBwolficeBwokeyBwokewokeBwokehhhhBwokehBwokeeoBwokeeeeeeeeeeeBwoiiiBwoeeeeeBwodyaBwoawoowmanaabBwlupunBwluBwlaupnB	wkwkwwkwkBwkwkwkwwkwkkwkBwkwkwkwkwkwkwkwkwwkwBwkwkwkwkwkwkwkkwkkwkwkwkwkwkkwkBwkwkwkwkwkwkwkkwBwkwkwkwkwkwkwkBwkwkwkwkwkwkBwkwkwkwkwkkwkwlwllwkwB	wkwkwkwkwBwkwkwkwkkwkwkwkwkkwkwkwkwkwkwksB&wkwkwkwkkwkwkwkwkkwkkwkwkkwkwkwkkwkwkwB
wkwkwkwkkwBwkwkwkkwwkkwBGwkwkwkkwkwkwkkwkwkwkkwkwkwkwkkwkwkkwkwkwkkwkwkwkkwkwkkwkwkkwkwkkwkwkwkwBwkwkwkkwkwkB	wkwkkwkwmBwkwkkwkwkwkwkBwkwkkwkwkwkwBwkwkkwkwBwkwkkkBwkwjwkBwktunyaBwkkwkwBwkenkwBwkakaB	wjsjsuwjsBwjshhaBwjhnyBwjhBwiwjsBwiwBwithinBwisssBwishBwisataBwiridanBwinnieBwinniBwinkBwinfowsBwindaBwimakomBwilujengB	wiirelessBwiiihhhBwiihBwignyaBwifiiBwidyaBwibBwhuteBwhuahahahahaBwhsuBwhozzBwhoB
whiteheadsBwhitBwhhhhhhhhhhhhhhhhhhhhhhhhhhBwheyBwhalBwhaaaaBwhBwfinyaBwewBwetB	wesssssssBwesssB	wessjosssBwessBwerlesBwenakB	wenaaaaakBwenBwelllBwelldoneBwelehBweifangBwehhhhBwehbB	wegdesnyaBwegdesBwefiBweekendBweekB
weeeetttttBweeB	wedgesnyaBwedgeesBwechatBwecastBwebnyaBwebbingBwdBwcBwbalBwaxvacBwaxingBwawasanB%wawamantappppnnnshdkshsndkshsjskssjjaBwavBwauuBwatterBwatnaBwatirB
waterproffBwaterfopB
wateresistBwaswasB	wasukrikaBwassalamBwashedB	wasaqonahBwasapBwarnyaBwarnnyaB	warnnanyaBwarnnaBwarninyaBwarnhBwarnetBwarnayaBwarnaskBwarnanyawarnanyaB
warnanyagkB
warnanyaaaBwarnanB
warnamerahBwarnaaaaaaaaaaBwarnaaaaBwarmsBwarmaBwarkopBwargaBwarehousenyaBwareBwarchBwarbiyasaakkBwarbiyasaaaahhhhBwarbiasaBwarbaBwarantyBwaranaBwarBwansonicBwanrnaBwannaBwanitaanBwangixB
wanginyaaaB'wangiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiBwangiiiiiiiiiiiiiBwangiiiiBwamaBwalwpunBwalpunBwalowpunBwalouBwalopnBwalkBwalhasilBwalhamdulillahBwalaupumBwalaupnBwalaunBwalaipunBwalafiatBwakwawB=wakwakwakwakwakwakwakwajwakwauwauwauwauwauwauwauwauwauwauederBwaktuxBwakruBwakeBwakdoyokB	wakakakakBwakakakaBwakainyaBwakBwaitBwaistbagBwaisbagBwahidBwahhhBwahhB	wahananyaBwahabBwaguBwagelaaaaaaBwafernyaBwaferBwafelBwaernaBwaduhhB	wadgesnyaBwadawBwadadawBwadadaBwachBwaawBwaahBwaaawBwaaahhhhhhhhBwaaaaB7vzgsvsvsvahbavacacavavavagacacgzvzgzvzvzvzgzvzgzvzvzvzgBvxxssBAvvzvzhzhzhxhxhdhhdjdudidhdhbdbdbdbdbjsjsjdjdjjdjxjxjxjxjxjdjdkdkdBvvvvgggggggBvvvcvvvctttB	vuiufsdjuBvsjsvakavsisbmavsosblfdfB	vsjscsjwnBvsjkwowpB-vshsjavdhdjsvxhsjsbxhsjsvxjshahajajakaoaoajsbBvsbsjsbvsbsjsbfdcBvpatBvoxterBvoviBvoteBvossBvolyBvolumenyBvoltricBvoltaseBvollyBvoliBvolaBvokalBvoerBvocalBvocakBvmcBvmaBvloomBvloggingBvlcBvkhplhjjjjjjjjnjjjjjjB
vjujkkgjlkB(vjfjfjfjdjdjdjxjfsjrhuhxhxjxjjdjfjxjcjcjBviuBviturBvitaminBvitaBvisualisasiBvistaB	visionnyaBvirexBvirekB	viralshopBvinshopBvignetBvigmetBviewingBvienaBvidoBvidionyaB	vidionnyaBvideoknBvibratornyaBvibratorBvibrBviberBvhnjkoksksosmsowowBvhfBvhahB#vgghvbnkjhgfvvbnnjhhhbbbnkkkkkkkkkkBvggggggghhhhhBvgfxghhB*vgbybybybtvtvtbtbtbtbtbtbtvtvrcectbunyvrvrBvfoB#vfdggfvhdxhysfhhtggdddthcfddfijcddfBverygoodBversinyaB	versatileBversaceBvernishBvermakBveraBveniamBvelvetBvelcroBvdeoBvdcBvcepatB-vcdgnzhdjfufududifitducjcjuikkvoooovvoyvvocivBvcdesrggthvddsbvddjB
vbxfvxdhccBvbnnnnnnnnnnB!vbbbvvbkkkjzjajjjajaiququywtwrqgaBvarietyB
variasikanB	variannyaBvariableBvarangBvapurBvaporBvapingBvapersBvanilaBvanesnyaBvalvetBvallenstoreB	valentineBvalenciaBvakumnyaBvaeiasiBvacBuyeeeeBuyBuwowwBuweBuvbBuuuuuuuuBuuuuuuuBuuuuBuuuhBuuuBuuhhhhkkkkkkkBuuhhBuuButunButlkButlButgButakBusvsveBusvshsbsjshsvsjshsvshsjsbsbsjsBusulBusirBusingBushshBushsbsusnsbsabasBuselessBusahayhBusahanyeB	usahannyaBusahamuBusahainnBusahainBusaahaB	urutannyaBurusBurungBurhBuratBuranBuraianBupsssBupsBupperBupilBupgradeBupdtBupayakanBupasBuoBunyuuuuuuuuuuuuuuuBunyungBunyBuntuntBuntunnyaBuntulBuntukygBuntukkB	untukanakBuntubgBuntilBunthkBuntgBunsurBunrecomendedBunrecognizedBunpredictableBunoBunntBunknowBunisexBuniqueBuniqBunikkBunguuBungunyaBunguhB	ungkapkanBungguBunfixedB
unexpectedBunegBuneB
understandBunderarmourBunboxingnyaBumuranBumtukBumrikBumpetinBumiB
umbuhannyaBumbrellaBumayanBumatBumamaBumahBulngBullamcoBulisnyaBulisanBulekBulasannBulasanmuBulangnyaBulalalllalllaaaalllalalaaaBulahBukyranBukutanB
ukuruannyaBukurqnBukurnnyaB	ukuranyyaB	ukuranpunB	ukuranntaB
ukurannnyaB
ukurannnnnBukurangB
ukulelenyaBukrnnyaBukrannyaB	ukrananyaBukhBukaranyaB	ujurabnyaBujungnyBujuBujkvlmBujjjBujanBuisBuihbkohvB!uigogkfificciccvkfivvkfifdxhfudssBuhuukkBuhuukBuhssuisdbduxjsosjsjeonwjsdBuhhhuuuukkkBuhhhuuukkkkkkBuhfBughhhhhghhgdBughhB4ugcyvhbjbjvycyfybyvrvtvhdufufhngfunfhfhdyfhfufydhfhdBufahBuenakB	uenaakkkkBueeennaaakkBuduBudsBudinBudejBudehBudaraBudangnyaBudangBudahvdiBudahanBudagBudaaB
ucjdhsydjvBuciBubBuawetBuaploadBuapikBualnyaBtytttrBtyjlugyoBtydackBtwwaBtwuBtwpatBtwnyBtwitterBtwiterBtwipwisBtwillBtwiceBtvnyaBtutupinBtutupiBtutsnyaBtusukanBtusukBturunnyaBturunanBturkishBturingBturbanyaBturahanBtunnerB	tunjuknyaBtunisiaBtuniknyaBtunguBtungfuinBtunerBtunedBtuneBtunanganBtumpukanBtumpukBtumpahanB	tumbuhnyaBtumbuhketagihanBtumblrB
tumblernyaBtumbaiBtumbBtulleB	tulisnnyaB	tulisanyaBtukuoBtukeranB,tukbvvklllkjjjkkkghllggjllhjkkkkjkjjhrfffgjjBtukaranBtujuhB	tujuannnnBtujuaBtujuBtugasnyaBtuekrBtudungB
tuchscreenBtubenyaBtubaBtuasnyaBtttuuuulllllllllllBttttuuullllllllllBtttttttyttjhB5tttttttttttttttttttttttttttttyyyyyyyyttttttttttttttttB6ttttttttttttttgyttttttttttttttttttttttttttttttttttttttBtttttrreerrrryttyttttB ttttrrrrrrrnnnnnnnnnnaaaasssaaaaBtttBttoopppBttombolBttlBttipBttehBttdBttappiiBtsrimaB
tschscreenBtryataBtruzBtruthaccBtrutamaBtrustB	trussssssBtrunB
trumakasihBtruckerBtrsumbatBtrsmkhBtrsegelBtrsdiaBtrrrbbBtrrrBtrrnyataBtrrimakasihB	trpercayaBtrozB	trnsparanBtrnsaksiBtrnBtrmxBtrmpatBtrmkhB
trminalnyaBtrmhBtrmakshBtrmakashBtrmahBtrlhatBtrlebihBtrlbihBtrlbhBtrlambatB	trlamabatBtrksturBtrkshBtrkesanBtrkBtrjaminBtripoBtripleBtrimzB
trimstrimsBtrimsssssssssssssBtrimssssssssssBtrimsssBtrimmsBtrimmmmmssssB	trimmmkssBtrimmerBtrimksiBtrimkasiiiiiiiihhhhhhhBtrimkasiBtrimkashBtrimkaihBtrimkB
trimaksiihBtrimaksihhhB
trimaksihhBtrimakkasihBtrimakasiiiB
trimakadihB	trimabkshBtrimaaciBtrikasihBtrikBtriimsBtriiimsB	triblenyaBtribelBtrianglenyaBtriamakasigBtrialBtrhambatBtrhadapBtrendyyyyyyyyBtrendBtrekerBtrekBtreatmenBtrdengarBtrdapatBtrcantumBtrbayarBtrbBtraumaB	trasparanBtrasferBtrasaB	trapeziumBtranstvBtransmittedB	translateBtransformernyaB
transaktsiBtransaksiknBtranningB
tranksaksiBtranBtraktirBtrakasiBtrailBtragediB
trackernyaB	trabsaksiB	trabelnyaBtqyuBtqvmBtquBtqqBtqpiBtqmbahBtqiuBtpuBtpouchBtpokBtpnB"tpitergantunglokasinyasihhhhhhhhhhBtpissBtpiiiBtparnaBtpakaiBtoysBtowerBtourengBtouchscrennB
touchscrenBtouchscreenyaBtoucBtotBtoskerinBtoscanyBtorsoBtorchBtorabikaB$toptoptoptoptoptorprotptotototootptpBtopsBtoppptoppppB@toppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppB9topppppppppppppppppppppppppppppppppppppppppppppppppppppppB+topppppppppppppppppppppppppppppppppppppppppBtopppppppppppppppppppBtoppppppppppppppBtopppppppppppppBtopppppppppppBtoppppppppppB
toppppppppBtoppoBtoplesBtopixBtopedBtopdehBtopcerrrBtopcerBtoopppppppoooofkfkfkfkfBtoopppppBtooppppBtooppmmaaaarrkkkooootttoooopppBtooppBtooopppppppBtoooopppppppB
toooopppppBtoooopppBtooooopBtooooooppppppppppppBtooooooppppBtooooooopppppppppBtooooooopppB	tooooooopBtooooooooooooopppppppBtooooooooooooooooooooooopB*toooooooooooooooooooooooooooooooopppppppppB0toooooooooooooooooooooooooooooooooooooooooopooopBtoolsBtonikBtongsisxB
tongkatnyaBtoneB
tonafkotopBtommyhilfigerBtommyBtomangB	tolonglahBtollBtolgB	toleransiBtoleranBtoledorBtolBtokonyBtokinaBtokiBtokenBtokcerrrBtokcerBtokanB	toiletrisBtoeflBtockcerBtobgpBtobatttB	tobaattttBtoaBtnytaBtnytBtnykanBtngsisBtngktBtngksB	tngkatkanBtngerangBtngannyaBtnganBtnfBtnaohsnxjxhnzozujaBtnamanBtmurBtmskBtmpilnBtmpatnyBtmpakBtmksiBtmhBtmbolBtmblBtmasukBtmaksihBtlitiBtlebihBtlapakBtkzBtkutnyaBtktBtkssssssBtksssBtknsBtkhsBtkhBtkgBtkdakBtkangBtjoyBtjhanksBtiyangBtiviBtiusBtitpisBtitikiBtitiBtitaniunBtitanBtitalBtitBtisunyaBtistisBtisakBtirusB
tirmakasihBtirexBtipuuuuuBtipuuuBtipuisBtipuanBtipstipsBtipsnyaB
tipistipisBtipisssBtipisnyaBtipislahBtipisanB
tipikalnyaBtipikBtipiissB
tipiiissssBtipiiisBtipiiiisssssBtipiiiiiiiiiiisBtipidBtipiBtioBtinjauBtingktB	tingkatknBtingkatkaaanaB	tingkatanBtingkanBtinghkatkanBtingglBtinggkatkanBtinggkatBtinggkanBtinggianB
tinggalkanBtingaliBtingaktBtingakatkanBtingakatBtindaklanjutnyaBtinaBtimingBtimeetimeemmmmmmmmmmmmmmmmB	timbulnyaB
timbanaganB	timakasihBtimahBtimBtiltBtilongBtikusnyaBtiktokBtigkatBtiggalB
tifissssssBtifakBtidurnyaBtiduranBtiddakBtidanB	tidaksukaBtidaksesuaiB	tidakbisaBtidakberjalanBtidakbegituBtidakaB
tidajvbaikBtidajBtibahBtibagBtibaaaaaaaaaaaaaaaaaaaaaaaaBthzB
thxzxxxxxxBthxyouuuBthxxxxxxxxxxxxxxxxxBthxxxxxxBthxuuuuBthxuBthxkBthusBthumbBthuBthreeBthreadBthoughBthornBthoBthnkyouBthnanBthkzBthkssssssssssssssssssssBthkssB
thinklightBthinBthicknesBtheyBthermalBtherionB	theraskinBthenkyuBthenkBthengkiuBthemeBthemBthekzBthaxxBthatsBthaterB
thansthansBthanssBthanlB3thankyouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuBthankyouuuuuB	thankyouuB6thankyoiiuiuuuuuuunsnssjsdkkdkxkcodkrkdodkjdodkdodjdkdBthankuB	thanksyouBthanksthanksB$thanksssssssssssssssssssssssssssssssB thanksssssssssssssssssssssssssssBthanksssssssssssssssssssBthankssssssssssssssBthankssssssssssB	thankksssBthankksBthankkksssssssssssssssBthankkksssaaaassBthankkkkkkkkkkkssssssssBthankkBthankdddsddsdasssdssBthangBthakBthaiBthaannkkBthaaannkksssBtguBtgnnyaBtgnBtglnyaBtgguBtggapBtggalBtfzB#tfvhghhhghhvbjgjcddghfhgfhvnfghfbhyBtftBtextureBtextileBtexpressBtewtetewtewtetetwBtewasBtetepiBteteangggaaaBtetdebutBtetaplahBtetanggapunBtetaiBtestyBtessssBtessBtesponB
tespacknyaBteslaBterytBteryantaB
teruuussssBteruusssBterussssssssssssB	terusssssBteruskannnnBterukurBterukrBterueBteruaBtersusunBtersisaB	tersinsalB	tersimpanB
tersendiriBtersematBterselubungBterselasaikanB
tersebutttBterseB	tersayangB
tersangkutBtersampaikanBtersamarkanBterryBterrrrrmurahBterrimaBterriimaakaasihhhB
terpuaskanB
terpontangBterpihatB	terpercyaBterpercayaaaBterpercaayaBterpengaruhB	terpencetB	terpecayaB	terpantauB
terpackingBterossBteroooshhhhBteroooooooossssssB	terombangBternyayaBternyatsBternyataahhB	ternyataaBterngkauB	ternayataBtermurahpengirimanBtermurahhhhhhhhhhhhhhhhhhhhhhhhBtermoBtermalB	termaksudBterlockBterlnjurBterlmbtBterllauBterlindungiB
terlindungBterlihanBterligatBterlewatBterletakB
terlengkapBterlauB	terlanjutB	terlampauBterlamatBterlamaB
terlaluuuuBterlaluencolokB	terlaaaffBterkrmBterkriimBterkoyakB
terkontrolBterkonfirmasiBterkocakB	terklupasBterkiniBterkerokBterkenalBterkemasBterkeletBterkejutnyaBterkecilB
terkangkauBterkabulBterjepitBterjatuhB
terjangkawBterjangkauuuuuuBterjangkauuuuB
terjangkaoBterjanBterjaminnnnnnnnnnnnnnBterjakauBterjakangkauB
terinstallB	terinstalBterinjakBterindikasiBterinciBterinakasihBterinaBterimskasihBterimsBterimqBterimmaaBterimaterimaBterimasiuiiiiijiiB
terimasihhBterimasBterimalasihBterimaksiihBterimaksihterimaksihBterimaksihhhBterimaksihhBterimaksasihBterimakishihB"terimakasiiiiiiiiiiiiihhhhhhhhjhjhBterimakasiiiiiiiiiiihhBterimakasiiihhhhhhhhBterimakasiiihhhhhBterimakasiiihhhBterimakasiiihhBterimakasiiiBterimakasiihB!terimakasihterimakasihterimakasihBterimakasihterimakasihBterimakasihhhhterimakasihhhhB"terimakasihhhhhhhhhhhhhhhhhhhhhhhhB terimakasihhhhhhhhhhhhhhhhhhhhhhBterimakasihhhhhhhhhhhhhhhhBterimakasihhhhhhhhhhhhhBterimakasigB	terimakasB
terimakaihBterimakaaihBterimaasihhhBterimaaksihBterimaaaaaaaaaaaaaaaaaaaaaBterimaaaaaaaaaaaaaaaaaBterimaaaaaaaaaaaaaBterimaaaBterimaaBterikmakasihBterikatBterikaBteriimaBteriamakasihBteriakBterhpusBterhmbatB	terhinggaB	terhindarBterhapusB	terhambatB	tergulungB
tergoncangBtergodaBtergerusBterehBterealisasiBterdetekBterdetecBterdekatBtercoverBterciptaBtercidukB
tercengangBtercakupBterbyataBterburukBterbukiB	terbuhungB
terbongkarB
terbohongiBterberatBterbeliBterbayarlahBterbayarkanBterbatalkanB	terbarunyBterbanginnyaBterbaiksssssssBterbaiksBterbaiklahhBterbaikkB
terbaikdahBterbaijBterbaiiiiiikBterbagusBterbagiB
terbaaaeekBteratrikBterapyBteranngBteranginBteramatB	teramasukBterakitB	terakasihBterahirBteraB	tepungnyaB
tepokjidatBteplokBteplekB	tepiannyaBtepattttBtentngBtenpatBtennisBtengsBtengokBtengkyuuuuuuuuuuuBtengkyuuuuuuuuBtengksB	tengkorakBtengkiyuuuuuBtengkiwBtengkiuBtenghBtenggorokanB	tenggelamBtenggangBtenderB	tendanganBtendangBtendaBtenannnnnnnnnnnnnnnnnnmnBtemuinBtemuiB	tempurungBtempurBtempuhB	tempperedBtemporBtempoeB
tempetaturBtemperaturnnyaBtemperatureB
temperaturBtempelinBtempeliBtempelannyaBtempatkuBtemmpatBtemhusBtemenkuBtemeninBtembemB
tembakonyaBtembakoBtemannyaBtelusuriBtelponanBtelpinBtelphoneBtelpanBtelornyaB	telitinyaBtelingannyaBtelihatBtelfnB
teleskopikB
teleponnyaBtelepasBtelaluB	telaaatttBtelaaatBtekukanBtekturBteksnyaB	tekendalaB	tejangkauB
teimakasihBtehnisiBtegorBtegapBtegangnBtegangBtegalBtegakBteeusBteelaluBteeimakasihBteeeriimakasiihhhhhhhhBteddyBtecticlBtechdooBtecamtumBtebslBtebetBtebeelBtebarBteballllBteballBtebalkanBtebalinBtebalanBtebakanBtebaaaalBtdyBtdtkBtdsBtdlBtdktauBtdknyaBtdkkBtdinyBtdhshfhdjgmfgshydhydBtcguBtbtbBtblBtbelBtbaBtayoBtayangannyaBtaxkBtawonnyaBtawaranBtawarBtauyaBtauuuBtaunBtauhidBtauhBtattoBtatoB
tatnafnfanBtatikBtatapiBtatakanBtatacaraBtassssBtasnyaaaBtasnyBtasngaBtaslarisBtasikmalayaBtasenyaB	tasbihnyaB	tarvelingBtaruhBtaroknyaBtarohBtaripBtarilamB
tarikannyaBtarifB	targetnyaBtarawihBtaraaaaaaaaBtaraaaaBtaraB+taptttttttttttttttttbbbbbbbbbbbeeeeeeetttttBtappppppppppppppppppppBtaplahBtapishBtapisBtapirBtapipackingBtapiiiiBtapiiiBtapihBtapibygB
tapibpuasaBtapiblafaznyaBtaperBtapatBtaonBtaoiBtanzxBtanyainBtanteuBtankzzzBtankzBtankyuBtanktopBtanknyaBtankdBtanjungBtangungBtangselBtangnyaB	tangngungB	tangkapanB	tanggungxBtangguBtangglB
tanggerangBtangganBtanggabBtanggaBtangapinBtanganyBtanganxB	tangannnnBtanganinBtangalB	tanduknyaBtancapBtanbahBtanamnyaB	tanamanyaBtamvanBtampilinB
tampilanyaB	tampilangBtampiBtampangxBtampangBtamengBtambalBtambaihBtambahannyaB	tambahannBtamagociBtaluBtalliBtalinyaaB	taksesuaiBtaksBtakeraBtakeBtakcobaBtakbiranB
takasimuraB
takarannyaBtakaranBtajmBtajaaaamBtailBtahuanBtahlilBtahilalatnyaBtahapanBtahannyaBtahankssBtahangBtagihBtadaBtadBtacappppBtabunganBtabmbahannyaBtabmannnnnnnBszB'syuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuukaBsyuuuuukkaaaaaakkkkBsyuuukaB	syukurlahBsyukonBsyukakkkBsyukakB	syukaakkkBsyukaaBsynkB,syipppppppppppppppppppppppppppppppppppppppppBsyipppppBsyiipBsyiiiipBsyibbbbBsygxB	syekaliiiBsyedihBsydahB	sybpengenBsyantikBsyangnyBsyangboxBsyahriniBsyahduBsyaangBsxanBswtB	swskeipsiBswrBswpeB	swpatunyaBswnangBswmuaBswlamatBswitctBswitBswichBswgituuuuuuuuuBsweeaterBsweatrB	sweaternyBswdapBswaraBswanBswallowBswabnyaBsuxesBsuwunnnBsuwooonBsuwoonBsuwiBsuwekkkBsuweeBsuvdobotBsuuuuggooooyyyBsuueeeBsutterBsutingBsusususususuausuuskakaakakkkakBsusukannBsusuiBsuskesBsushBsusesBsusahaBsuryaBsurveyorBsurveyBsuruBsursBsurroundBsurprisinglyBsurpriesBsurgaBsuratmanBsuratapBsuranyaBsurantaplahhBsurantafBsurantabBsupriseBsuppriceBsupppoBsuppperrBsupposedBsupplyBsupplierBsuppBsuplayerBsuplayBsuplaiBsuperrrrrrrrB	superrrrrBsuperrrrBsuperrrB
supermurahBsuperflashdealB	superdealB	supercarsBsuperantappBsuperaccBsupeeerBsupeeeeeeeeerBsuoerBsuoBsuntikanBsuntikBsunglassesnyaBsungkanBsungBsundulBsundahBsunBsumutBsumpahinB	sumpahhhhBsumpahhBsumpaahBsumpaaaaaaaahBsumedangBsumbrBsumbiBsumberBsumawitiBsulsesBsulitnyaBsulitkadangBsulamiBsukssBsuksezBsuksessssssssssssssssssB	suksessssB	sukseskanBsuksenBsuksemaBsukselahB
sukseeesssBsukseaBsukseB	sukoharjoBsukkkkaaaaaaaaaaaaaaaaaaaaaaBsukkkaaaaaaaB
sukkaaaakkBsukkaaaaBsukkaaBsukessBsukesBsukcesB sukasukasukasukasukasukasukasukaBsukasukajcjcuccjckvkvivkcBsukaresponyaBsukarB(sukandkddnfnfckccpckfnfnccjccbbcjckcncjcBsukancptBsukanBsukalahBsukakkkkkkkkkkkkBsukakkkkkkkkkBsukakkkkkkkkB	sukakkkkkBsukakkkkBsukakkkBsukakkjBsukakkBsukaeunBsukaessssseeBsukaesB
sukabangetBsukaalahBsukaakkB	sukaaakkkBsukaaakkBsukaaaasukaaaaB	sukaaaaakBsukaaaaaaaasukaaaaaaaaaaBsukaaaaaaaaaaaakkkB3sukaaaaaaaaaaaaaaaaaaaassaaaaaaaaasssssaaaaaaaaaaaaBFsukaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaakkkkkkkkkkkkkkkkkkkkB(sukaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaB sukaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBsukaaaaaaaaaaaaaaaaaaaaaaaaaaaBsukaaaaaaaaaaaaaaaaaaaaaaaaaBsukaaaaaaaaaaaaaaaaaaaaaaaaBsukaaaaaaaaaaaaaaaBsukaaaaaaaaaaaaaBsukaaaaaaaaaBsujaBsuiterBsuiteBsuitableBsuitBsuippB	suiipppppBsuiipBsuiiipppBsuhunyaBsuguhinBsuerBsuegerB
sueeeeeeeeBsueeeeBsueeBsudshBsudjaBsudhlahBsudaqhBsudangBsudakBsudahsampaiBsudahsBsudahnyampeBsudahnaampeBsudahhhhBsudahhBsudahblBsudabBsudaahBsudaaaahBsudaaaaaaaaaaaaaaaahhhhhBsuchBsucessBsuccesB	subwooferB	subscribeB	subjektifBsubhanallahBsubB	suarannyaBsuarakuBsuaraaBstylistBstylingB	stutusnyaBstufB	studionyaBsttsBststusBstrumnyBstrongBstronBstripingnyaB	stringnyaBstringBstretchyBstretBstressBstreetchBstrechiBstreatchBstreamerBstreamB	strawberyBstoreeeeBstorageBstopperBstopnyaB
stopkontakBstonesBstoneBstokingBstockingnyaBstnghBstndaBstnBstmBstjoperBstimulanBstilyBstikntaB	stidaknyaBstidakB
stickernyaBstghBsterioBsteplessBsteplesBstengahBstenbayBstelBstekBsteamnyaBsteamB	steadicamBstdknyaBstatusyBstationBstarapBstaplessB	staplesnyBstaplerBstansdaBstanleyBstandoutBstandnyaBstandingB
standarnyaBstandarisasiBstalahBstagnanB	stabilnyaBstabilloBstaaandaBssyyBssuiBssuailahB#sssssssssxxxxdsssssssssssssssssssssB)sssssssssssssssssssssssssssssssssssssssssBOsssssssssssssssssssssssssssasalllllllllllllllllllllllllllllllllllllllmmmmmmmmmmBssssssssssssssssssssssssssBsssssssssssssssssssssssBssssssssssssiiiiippppppB
ssssssssssB!ssssssiiiiiiiipppppppppppppppppppB(ssssssiiiiiiiiipppppppppppppppppppppppppB0sssssiiiiiiippppppppppppppppppppppppppppppppppppBsssssB"ssssiiiiiiiiiiiiiiiiipppppppppppppBsssjdjbjhshsbsvdbdhdhddddBsssiiiiippppppppppBsssddddddddddddBsssBssksldB	ssiiiiippBssiBssesuuaiBssesuiB
ssedangnyaBssdBssbelahBssatBssampaiBssaiBssahBsryBsrviceBsruhBsrnkanBsrmogaBsrlamatBsringBsrgBsrekBsreeegggBsranBsqmapaiBspycamBspunbondBsptunyBsptnyBsprayerB
spppppppppBspoonBsponsorBsponsBsponnyaBspongeBsplitternyaBspliterBsplitBspleBspiritBspionnyaBspinningB
spinnerngaBspinetB	spidermanBspesifilasiBspesifikasinnyBsperifikasinyaBspektakulerrrrrrrrrrBspektacularBspeksifikasiBspekndanBspekernyBspekearnyaaBspekearBspekaerBspeedyBspeednyaBspecsBspeakerxBspeakersB	speakernyBspeakBspdB
spatulanyaBspatulaBspasifikasiBspasangBsparpaBsparingBsparepatBspaekerBspaceBsowakB	souvenierBsourceBsounessB	soundcardBsounBsotexBsossssBsosialBsosbsosbBsorrryBsorongBsoppengBsooBsonnyBsongnyaBsongketBsondBsomplakBsomogaB	sometimesBsolutionBsolokBsolehahBsolehBsoldoutBsolderanBsolatipBsolanyaBsolanBsoketnyaBsoketBsohBsofwareB	softtwareBsoftlensnyaB	softlenseB	softjeansBsoftexBsoftblueB
soflentnyaB	soflennyaBsofcaseBsoeknyaBsodaqallahuladzimB	sodakohinBsocketBsobeknyaBsobekkkkkkkkkkkkB
sobekannyaBsobbB
sobattttttBsobBsoanyaB	soalnyadlBsoalnuaBsnpeBsnorkelBsnnangBsniperBsnieeeBsnieBsngguhBsngarBsnengBsnapiBsnalpotBsnagtBsnagatBsmwBsmurfBsmuannyaBsmuahBsmschatBsmruputBsmriwingBsmpsBsmpkBsmpinyaBsmpeiBsmpeeeBsmpayBsmpaoBsmpaitBsmpaikanBsmpaiiiiiiiiiBsmpaeBsmpaaiBsmoogaBsmokeBsmogahBsmofsBsmoBsmntrBsmntaraBsmngatBsmllBsmknBsmgaaBsmentaraBsmellBsmdriBsmcamBsmbuhBsmapiBsmanyaBsmampeBsmalamBsmaapiBsluBslsaiBslowresBslottBslmattBsliverB	slitingnyBslimsBslimnyaBslimeBslhslhBslettingBsletinqB	sletinginB
sletengnyaBslengkapBslempangnyaB
slempanginBslemanBslekBslbmnyaBslbihBslawBslaowBslaluuuBslalomBslainBskyBsksnsksbakanaianjbsBsksjbsbahhshsBsksBskrupnyaBskrupBskopBskolahB
skmeiskmeiBskmeiBsklhBskechersnyaBskatenyaBskalaBsjzghxsBsjsjsjakakansbbsbabaBsjsjdjdjzjzjajB)sjsjdhdjsidjcnjdjcusicjcjjscijsudvuhfofjfBsjshgshaczusB0sjshevanskgakajgabwjajajqkakabvahajajavahafavacaB	sjsbsnsshBsjjhisjsvuvsvysvysvsysBsjcamBsjajsgsakkabsvsBsjaaBsizengaBsiyapBsixBsitunyaB
situasinyaBsituasiBsisuaiB	sistimnyaBsistimB
sistematisBsistanyaB	sistahnyaBsistaaBsisssBsisbdksoBsisanBsirupB	sirkulasiBsirikBsirihBsipzBsiptttBsipthxBPsipssssssssssssssddddddddddsssssssssssssssssssssssssssssssssssssssssssssssssssssBcsipsispsiaoaooaoaooaoaoaoaooaoakakaakkaakakakkakakaaakakajajjajajajaajjajajajajkaakaakkaaaaakaaaaaaB0sipsipsispispsispsispisosispsispsipsospsispisosoBsippsssBRsippppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppBFsippppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppB4sippppppppppppppppppppppppppppppppppppppppppppppppppB/sipppppppppppppppppppppppppppppppppppppppppppppB*sippppppppppppppppppppppppppppppppppppppppB(sippppppppppppppppppppppppppppppppppppppB'sipppppppppppppppppppppppppppppppppppppB sippppppppppppppppppppppppppppppBsippppppppppppppppppppppppppBsippppppppppppppppppppppppBsippppppppppppppppppppppBsippppppppppppppppppBsipppppppppppppppppBsipppppppppppppBsipppppppppppBsippppppppppooooooooooBsippppppppoBsipppppppokeokwokwBsipppppooooooooooooooBsippplahBsippoppppppppBsippoBsippdehhBsipokehBsiplinBsiplahhhhhhhhhhhhhBsiphhhBsipenerimanyaB
sipenerimaB	sipemilikBsipelBsipangBsinzuiBsinylBsinyaldiBsinyajBsinyaBsinuBsinisBsininyaBsiniituBsinihB
singletnyaB
singketnyaB
singkatnyaB	singarajaBsinemaBsinarnyaBsimurahB	simpulkanBsimplleeBsimplleBsimpleksBsimpenyaB	simpelnyaB
simpatinyaBsimpatiBsimpananBsimerakBsimanisBsimakBsilveryBsillingBsilinderB	silikanyaB
siliconnyaBsiliconenyaBsiliconeBsilicaBsilentBsileletBsilahhhhhkannnnBsilagkanBsilBsikutBsikatnyaBsijoBsiiupBsiipssBsiipppppppppppppppppppppppB
siipppppppB	siippppppBsiipppllB	siippplahBsiiplahBsiiopppppppppB
siinginkanBsiijjpBsiiiupB"siiippppppppppppppppppppppppppppppBsiiipppppppppBsiiippplahhBsiiiopBsiiiipppppppB
siiiiplahhBsiiiiippppppppppppppppppppppppBsiiiiipppppppppppBsiiiiippppppppooooooooB	siiiiipppB siiiiiipppppppppppppppppppppppppBsiiiiiipppppppppppppppppppBsiiiiiippppppBsiiiiiipppppBsiiiiiippppBMsiiiiiiipppppppplllppppllllppppppppppppppplllllppplpppplpppppppppppppppppppppBsiiiiiiiipppppppppppppppppppBsiiiiiiiiiipppppB*siiiiiiiiiiippppppppppppppppppppppppppppppBsiiiiiiiiiiipppppB3siiiiiiiiiiiippppppppppppppppppppppppppppppppppppppBsiiiiiiiiiiiiippppppppbarangBsiiiiiiiiiiiiippppppBsiiiiiiiiiiiiiplahB'siiiiiiiiiiiiiiiiipppppppppppppppppppppBsiiiiiiiiiiiiiiiiipBsiiiiiiiiiiiiiiiiiiipppppppppppBsiiiiiiiiiiiiiiiiiiiplahBsiiiiiiiiiiiiiiiiiiiiiiipBsiiiiiiiiiiiiiiiiiiiiiiiiiiipBsiiiiiiiiiiiiiiiiiiiiiiiiiiiipB3siiiiiiiiiiiiiiiiiiiiiiiiiiiiipppppppppppppppppppppB"siiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiipB$siiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiipB'siiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiipB[siiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiipppppppppppplppppppppppppppppppppppppppppppppBsiiiiiiiiiiiiiiiiB	siiiiiiiiBsiiiiiiBsiiiiBsiifBsiichBsiicchhBsiiapBsihtapiBsihhhjjjhjjBsihhhhhjjjjjjBsiharBsignyalBsigalinggingBsifatBsifarBsieplahBsichhhBsicepotBsicBsibuknyaB	siaranpunB	siapsiappBsiappppppppBsiapinBsiapapunBsiaoBsianakBsiamBsialnyaB	siallaganBsialanBsialBsiagaBsiaaalBshuttlecockB
shutternyaB	shutternyBshureBshukronBshuffleBshueBshsuahshshsB'shssbhshshshsshshushddhhdhshdhhdhdhhdhdBshskakjahsgsB2shsjjsjsjssjjsjshdhdhdjdhdjdjdjdjsjsjsjsjsjsjjddjdBshsjdbdjskaBshshshsBshshshBshsbsBshsBshrusnyaBshrusnyBshrusBshrsnyaBshriBshouldBshotBshootingBshooterBshoopingBsholiBsholehBshokBshohokuBshockBshnggaBshkwkrkeBshizBshiseidoBshipmentBshipingBshiningBshiiipBshiiiitBshiiiiipBshieldB3shhshshshshshhshshshsyhehehehehehehehehehehehheheheBshhhhhhsBshggB	shekernyaBsheeranBsharrBsharingB	shapelessBshalBshakrulkhanB
shakernyaaBshaftBshadowwBshadenyaBshadaqallahuladzimBshabbyBshaaBsgtuhBsgthuBsgnnagnagnagBsgnBsgituuBsgbBsgatBsgalanyaBsgalaBsgaBsgBsfutfhhfhjfhjdghhgjidgihchjhbjvB0sffrfghgdfjhffhjgfbbjjjgfbgfchjyfdrghjnddssdfhnmBsexiBsewuaiBsewkbdBsevenB
sevelumnyaBsevBseupritBseumurBseumpamaB<seueeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeerrBsetypeBsetupboxBsetujuanB	settinganBsetrumBsetrlahBsetokBsetinggiBsetingannyaB	setidanyaB	setidaknyBsetidakBsetetesBsetetB	setepatanBsetempatBsetellahBsetelehBsetelatB	setelapakB
setelannyaBsetelanB
setelahnyaBsetelahbsayaBsetelagB	setaralahBsetanBsetabilBsesuweiBsesuweBsesuuuuaiiiBsesuuBsesutuBsesusB	sesuilahhBsesuiiBsesuiaaBsesueiBsesuayyyyyyyyBsesuayiBsesuauiBsesuauaiBsesuasiBsesualuiBsesualBsesuajBsesuaiygB	sesuaitasB
sesuaispekBsesuairesponBsesuainyangBsesuailhB	sesuailanBsesuaikanlahBsesuaiiiiuiB)sesuaiiiiiiiiiuiiiiiiiiiiiiiiiiiiiiiiiiiiBsesuaiiiiiiiiiiiiioooiiiiB3sesuaiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiB)sesuaiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiB sesuaiiiiiiiiiiiiiiiiiiiiiiiiiiiBsesuaiiiiiiiiiiiiiiiiiiiiiiiiiBsesuaiiiiiiiiiiiiiiiiiiiiiiiiBsesuaiiiiiiiiiiiiiiiiiiiBsesuaiiiiiiiiiiiiiiiiiiBsesuaiiiiiiiiiiiiiiBsesuaiiiiiiiiiiiBsesuaiiiiiiiiiiBsesuaiiiiiiiiiBsesuaiharganyaBsesuaidenaganBsesuaibtidakBsesuaibpesananBsesuaibdenganB
sesuaibdanBsesuaibB#sesuaaiiiiiiiiiiiiiiiiiiiiiiiiiiiiiB	sesuaaiiiB	sesuaaaiiBsesuaaaaaaaiiiBsesuaaaaaaaaiiBsessionBsesotBsesnsasiBsesikuBsesibukBsesiaiBseseuiBsesesuaiiiiiiiBsesegituBseseaiBsesauiiBsesampaiBsesaliBsesakliBsesaiiBsesaatB	servisnyaBserviseBservisanBservicnenyaB	servicingBservicesBserviceableBserverBseruuuBserutBserumxBsersakBserpihanB	serpentinBseriuuuddddsssBserinhB	seringnyaBseringinBseringanBserieB
seribuwwwwB	seribuuuuBserendahB	serencengBserehBseregBserebuuBserbuBserasoftBserapiBserangkaianBserakBserahkanB	seragaminB	serabutanBseraBsepwBseputarBseputBseptnyaBseppppBsepplahB
sepokatnyaBsepikerBsepiB	sepetunyaBsepetuB	sepetinyaBsepesifikasiB
seperempatB
sepekernyaBsepekB	sepedahanBsepeakerBsepbarangnyaB	sepatynyaBsepatuyaBsepatuyBsepatunysepatunyB
sepatunyaaBsepatunyBsepatunaB	separunyaBseparuhBsepanBsepakatBsepakBsepahaB
sepadanlahBsepadaBseowBseogaBsenyuminBsenyumBsenyapBsenuanyaBsenuaBsentimeternyaBsentilanB	senteenyaBsentausaBsentBsensuiBsensualBsensornyBsensitivitasnyaBsensitivitasB	sensitiveBsensitipBsensdiriBsenonohB
sennheizerB	senkernyaBsenggolB	sengajaanBsengaiB	senengnyaBsenenginB
sendirinyaBsendirikBsendiriiBsendiiriBsendatBsendalnxBsendalkuBsenasibB
senantiasaBsenanrBsenanngB
senangnyaaB	senanglahBsenanagBsemwanyaB	semurahanBsemulaBsemuaxBsemuanyyaaaaaaaaaaaaaaaaaaB	semuanyaaBsemuanuaBsemualahBsemuaaaaaaaaaaaaaaaaaBsemuaaaaBsemuaaBsemriwingggB	semrawangBsempwtB*sempuuuuuuuuiiiiiiuuuuuuuuuuuuuuuuuuurrrrrBsempurrrrnaaaaaBsempurnakanBsemptB	sempitnyaB
sempiiiittBsempgaBsempenBsempeBsemooogaaaaaaB%semooiooiioooiiiioioiioooigaaaaaaaaaaBsemongaBsemoigaBsemogahBsemogabermanfaatBsemogBsemoaBsemmogaBsemisalB
semingguanBseminggiBsemfitBsemeruBsemerekB	sementataBsemenjakBsemenaB	sembuuhhhBsemburannyaBsembronoBsembilanBsemberrrBsemauaBsemauBsemanisBsemangiBsemalattB
semaksimalBsemakinsuksesBsemBselusinB
seluruhnyaBselumBselulerBseluasBseluBselsluBselpieBselowwBselopBselmatB	sellwenyaBsellllllllllllllllllllllBselllerrBsellingBsellesBsellerrrrrrrrrrrrrrrBsellerrrrrrrBsellerrrBsellerrBsellernyaaaaaaaaBsellernyB	sellerlahBselleerrrrrrBselleerrrrrBselleeerrrrBselleeeeerrrrrrBselkerBselipanBselfinyaB	selfienyaBselfienyBseleseBselersBselerrrBselerrB	selerekanBselerakuBselerahBseleraaBselengkapnyaB	selengkapBselektifBselebihxB	selebihnyB	selebaranB
selayaknyaB	selatunyaBselarasBselaowBselanjutBselangitBselametBselamattBselamatselamatBselamaattttB	selalulahBselalluBselainyaBselaainBsekuttBsektorBseksyBseksiBseksehBsekrupB	sekringyaBsekrangBsekopnyaBsekopB	sekomplitB
sekolahnyaBsekolaB
sekitarnyaB	sekiranyaBsekiraBsekipingnyaBseketikaBsekeraBsekerB
sekelasnyaBsekejapBsekardusBsekaramgB	sekampungBsekaloBsekalitidakB	sekalinyaB	sekalinhaB.sekaliiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiB*sekaliiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiB"sekaliiiiiiiiiiiiiiiiiiiiiiiiiiiiiBsekaliiiiiiiiiiiiiiiiiiiiiiiiiBsekaliiiiiiiiiiiiiiiiiiBsekaliiiiiiiiiiiiiiiBsekaliiiiiiiiiiiiiiBsekaliiiiiiiiiiiBsekaliiiiiiiBsekalihargaBsekalihBsekalehB
sekaleeeeeBsekaleeBsekalaiBsekaalliiiiiiiiiiiiiiiiBsekB
sejujurnyaBsejernihB	sejengkalB
sejelasnyaBsejalanB
sejadahnyaBsejadahB	seinyalnaBsehrsnyaBsehrnyaBsehriBsehnggaB	sehingggaB	sehinggakBsehiggaBsehelaiB	sehausnyaBsehausnyB	seharusnyB	seharunyaB
seharihariBsehalusBsehabisBsegutuBsegtoBsegityB
segituuuuuBsegitusegituBsegituhBsegitigaBseginiseginiBsegereBsegellBsegarpengirimanBsegarisBsegaaaarrrrBseftiBsefantastisBseeuatuBseeuaiB
seeratusanBseeppppppppppppppppppppppppppppBseepppoBseenBseemsBseelasBseeeppppBseeepBseeelllllleeeeerrrrrrrrrrrBseeekaaaliiiiB)seeeelllllllleeeeeeeeeerrrrrraaaaaaaaaaaaBseeeeeeppppBseeeeeepBseeeeeeeeeepBseedBsedusBseduniaBsedowBsedoBsednagBsedktBsedkitBsedituBseditiBsediktBsedikitiB
sedikiiiitBsedikiBsedikBsediikitBsedianBsedeyanBsedekitB	sedangnyaBsedakahBsedB
secukupnyaBsecuilBsectionBsecoundBsechBsecenganBsecengBsecarikB	secangkirBsecaliBsebytkanBsebutkanB
sebsebesarBsebrpBsebotolBseblmnyaBseblmBsebersihB	sebenrnyaBsebenereB	sebelumyaBsebelumxBsebelomBsebelaahBsebegituBsebegaiBsebeberBsebbBsebarkanBsebarBsebandinganBsebandelBsebalikBsebalBsebaikxB
sebaiknnyaBsebagiB
sebagainyaBsebabnyaBseaueyBseauaiiiiiiiiiiiiBseauaBseasuaiBseandaiBseamuaBsealiB	seadanyaaBsduahBsdraBsdrBsdpeBsdotannyBsdnyaBsdngBsdmogaBsdkiitBsdiliatBsdhinggaBsderhnaBsdengBsdddfsdfffffffB
sdahsampaiB	sdahlahhhBsdahlahBscrubB	scrollnyaBscrollBscrewdriverBscratchBscoBscmBsclarBscientificnyaBschtBscarletBscarfnyaBscanerBscaleB
scabiesnyaBsbzjaknBsbtrB\sbssjsksksnshsjsksksbsbshsisididhdhdhhdhdjssknsbshsjsisksjshddudiisishshshsjsjsjjsjsjshhshshBsbrapaBsbrBsboumBsbnerBsbmBsblomBsbgmnBsbelonBsbeelumBsbandingBsbaikB
sbagaimanaBsazanamiB	sayyyyaaaBsaysayaBsayoBsayhBsaygBsayatBsayasukaB	sayasudahBsayasayaBsayapunBsayanlBsayangnyB	sayangmyaBsayangiBsayangeB
sayamngnyaBsayaangBsayaaaaaaaaaaaaaaaaaaaaBsayaaaaBsawoBsawanganBsawahBsaverBsaveBsatuuB	satupaketBsatuhariBsatugaBsatubyaBsatseutBsatpamB
satisfieldB	satisfiedBsatinnyaBsatenyaBsateliteBsatBsasetanBsasakB	sarungnyaBsaringannyaBsarapanBsarankuBsaragihBsaraBsarBsaptuBsapphireBsapmpaiBsapetBsapekkkBsapeBsapatuBsapatBsapariBsaodaraBsanyangBsanyaBsantunBsanpeBsanpaiBsankenBsanjayBsangtaBsangstBsangkingBsangitB	sangeeeetBsangattttttttttttttBsangattttttB
sangatttttBsangantB
sangaattttBsangaaatBsangaaaatttB	sangaaaatBsangaaaaaaaaatB&sangaaaaaaaaaaaaaaaaaattttttttttttttttBsanengBsandiagaB
sandangnyaBsandalxBsanbilBsanatBsananyaBsanahBsanaganBsanBsamuraiBsamsakBsampyBsampungBsampulBsampsiB	samprknyaBsamppeBsampinyBsampingxB	sampinganBsampeyaBsampeyBsampexBsampeuBsampesampeeB
sampenyaaaB	sampenyaaB	sampeknyaBsampeinBsampanB	sampakeunBsampakB	sampajnyaBsampaisampaighhhjjjjjjjjjjjjjBsampainyasampainyaB
sampainyaiBsampainyaaaaaaaaaaaaaaaaaaBsampainyaaaaBsampainyaaaB
sampainyaaB
sampainnyaBsampainB	sampaimdhBsampaimBsampaiiiiiiiijjjjB"sampaiiiiiiiiiiiiiiiiiiiiiiiiiiiiiBsampaiiiiiiiiiiiiiiiiiiBsampaiiiiiiiiiiiiBsampaayyyyyyyyB	sampaaaaiBsampaaaaaaaaaiiiiiiiiiiiiBsampaaaaaaaaaaaiBsamoaiB	sammmmpaiBsammaB	samlaiiiiBsaminBsamedayBsamdayB	sambunginBsambuganBsambilanB	sambelnyaBsambelBsambalnyBsambalBsamatB
samasekaliBsamasaBsamapaaaaaiBsamaiiiiBsamaiB	samadekerBsamabentuknyaBsamaanBsaluutB	sallernyaBsalleeeeeeerrrrBsaljunyaBsalinBsalepBsalemnyaBsaleerBsaleeBsalatigaB	salahsatuB
salahsalahBsalahinB	salahhhhhBsalahhhBsalahaB	saklarnyaBsakiyBsakitnyaBsakieuBsakepppBsakalaiBsakBsajalahBsajalaaaaahhhhhhBsajaaaahB"sajaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBsajaaaaaaaaaaaaaaaaaaaaaaaBsajaaaaaaaaaaaaaaaBsajaaaaaaaaB	sajaaaaaaBsajaaBsajBsaizBsaiyaBsaipaiB
saingannyaBsaikkkkBsahurBsahhBsahaB	safetynyaBsadulurBsadeanBsadddBsadaiBsadBsacaBsabunyaBsabunnyBsabukB	sablonyaaBsaberBsabarinBsabanariBsabaaaaaarrrrrrrBsaatvdiBsaatsaatBsaampeBsaampaiBsaamaB&saaaaaaaaaaaaaaaaaaangaaaaaaaaaaaaaattBrysakBrxBrwsponB
rwcomendedBrvcaB4ruuuuuuuuuuuunnnnnnnnnnniiiiiiiiiinnnnnnnnnnggggggggBruteBrusakkkkkkkkkkkB	rusakkkkkBrusakkBrusaaakkkkkBrupoooB	runcinganBrumiBrumbaiB
rumayanlahBrumaayanBrumB
ruksakkkkkBruginyaBrugiiiiiBrugiiiBrudetBrubyBruasBruarBruapiiiiBruangnyaBruangaBruamganBruBrssponBrsponyaBrsponxBrspihBrsmBrskBrsakBrrusakBrrruuuuaarrrrBrrfaysgwiwwitsipwohzgxbzBrresponBrrcommenfedBrqpihBrpntBrpengirimanBrosulBrostaBrossBrosokanBrosegoldBrontoknyaaaBrongsokBronggaBrongarB	rombonganBromBrolloonBrollingBrollerskateBrollanBrolexBrolBrokxBrokomendBrokemendBrohmatBrohaniB	rofundnyaBroecomendedBrodoBrodjatvBrodjaBrodaxBrockwollB
rocemendedBrobyBrobotBrobekanBrobeBrobalBrobahBroBrnyaBrnkBrnakBrmbtBrjinBrjaB rizjexxxxxxxxxxxxxxxxxxxxxxxxxxxBrizekBritzluitingBritnyaBrisrusB	risletingBrisihBriserBrisekBripitBripcurlBripBrioBrinsoBrinjaniBringyaBringnyBringihBrilakumaBrikuesBrikomenBrijekanBrijecBrightBridhoBridgenyaBridgeBricheseBribuuuuuuuuuuuuBribuuuuBribuuBribenBribeeeetBribahBribBriBrheheBrgbBrgaBrfBrezqiBrezponBrezkiB
rexomendedB	revommendB
revomendrdB
revomendedBreunyekBreturrrBreturnyaBretureBreturdiB
returannyaBretsluitingnyaB
retslitingBretsletingxBretsletingnyaBretsletingmyaB
retseltingBretrurBrestarBressponBresslerB	ressellerBresselernyaBresresletingBresppnBrespoudBresposnsBresposBrespoonnBrespoonBresponyaterimaBresponyB	respontyaB
respontnyaBresponsterimaB	responsivBresponsifitasBresponsibelBresponshiipB	responponB
responnyaaBresponnnnnnnnnnnnnnnnnnBresponnnnnnnnnnBresponnnnnnB	responnnnBresponnBrespondvokeBrespondrespondB
respondnyaB
respondingBresponcepatBresponceBrespomyaBrespomantappBrespobBrespnBrespinyaBrespinsBresopBresonBresommendedBresoluaiBresnyaB
resluitingBreslitingnyaBreslettingnyaB
reslettingB	resletinhBresletingnyBresletingngBreskonBresiverB	resistantBresipunB
resinyapunB
resinyanyaBresinBreshopBresfonyaBresfondBresfhonBresetlingnyaBresetingB	reseltingBresellerrrrBresellernyaBresellBreseleurB
reselernyaBrescpondBresahletingBresBreqwesBrequetBrequesyB	requestedBrequesB
reqomendedBreputasinyaB
repurchaseB
reproduksiBrepresentatifBreponsBreplenishingBreplaceBrepingBrepairB	repackingBrepackBreotB
reommendedBrenyakBrentaB
rensponnyaBrensponB	renponsifBrenovBrennyekBrennnnnnnnnnnnnnnnnBreniBrendanyaB
rencananyaB
remuukkkkkBrempeyekBrempelannyaBrempahBremotenyBremommendedBremocemB	remmondedBremixerBremendedBremaxBremasBremangBremahanBremahBrellBreliableBrelatipBrekuesBrekturBrekselBrekreasiBrekorB
rekomwndedBrekomrendedBrekomondasiB	rekomndedBrekommendetB	rekomenttB
rekomentedBrekomentB
rekomennnnB
rekomenlahB	rekomenedBrekomendesssssB
rekomendesB
rekomenderBrekomendeedBrekomendedsellerBrekomendedddddddBrekomenddddddddddfffffdbbbbbbbbBrekomendasjBrekomendasilahB
rekomendadBrekomeeendedB	rekomedetB	rekomededB
rekomedasiB
rekomandedBrekamnyaB	rekamankuBrejekBreilBreguBregoneBregisterB
regenerasiBreganeB
refreshingBreflikaBrefillanBrefereeBreferbushedBreelnyaBreelBreeBredponBrednyaBreddyBreddupdtmdsgiyiuououpupuippBredbullB	recordnyaBrecordingnyaBrecordedB
reconendedBrecomselB
recomondedBrecomntdBrecomndB	recommentBrecommendidBrecommendetBrecommendesBrecommenderBrecommendefrecommendefBrecommendedddB
recommendeBrecommenBrecommemdedBrecommdB
recomentedBrecomensellerBrecomenndennBrecomenndedBrecomendendB
recomendefBrecomendeeeeeeeeeBrecomendeeddBrecomendedtopBrecomendedrecomendedBrecomendeddddddddddjdjdbdbdhdhBrecomendedddddddddddddddddddddBrecomendedddBrecomendderBrecomendationBrecomendasikanBrecomendasiinBrecomeendedB
recomeddedBrecomedddddddddddddddddddddddB
recomedasiB
recomebdedBrecomdsellerB
recomdddddBrecomdBrecomandasiBrecomanB	recognizeB
recodernyaB	recmendedB
recimendedB
rechargingBrechargeB
recemendedBrecehanBrecehBreccomendesBreccomBreccimendetBrebuanBrebongBrebetBreaspontBreasponeBreasponB
reasonableB	reaponnyaBreapondBrealpictureBreallBrealitasB
realitanyaBrealitaaB	realistisB	realisasiBrealficBrealaccBreakasiBreaderBreaddyBreactionB	reachableBrcvrnyaB
rcommendedBrbbBrawisBrawatB	rawabuayaBrattingB	ratingnyaBrasulBrasponBrasionalBrasioBrasaxBrasannyaBrasanaBrasBrarapihBrapukBrapuBrappiihBrappiBrapohBraplikaBrapkBrapiterimakasihBrapirBrapijB	rapiinnyaBrapiiiiiiiiiiiBrapiiiiiiiiB
rapiiiiiihB
rapiiiihhhBrapiiiiB	rapiiihhhBrapiiihhBrapiiihBrapiihhBrapihpackingBrapihiinBrapihhhBrapihhbBrapihanBrapihaBrapidBraphiBrapaihBrapaBraonoB
rantingnyaBrantingBranteBrantangB
rantainnyaBranjangBrangkapB	rangkaianBrangkaBrangenyaBrangBrancakBranamanBrampeBramiBramhBrameeBramdomBrambahBramayanaBramaiB
ramahramahBramahhhhBramadhonBramaaahBramaaaaahhhhhhhhhhhhhhhBramaBralinelektronikBraliB	rakyatpunBraksasaBrakitBrakB	rajutanyaBrajetBraipBrainnyaBraincovernyaaaBraincoatBraihBraiderBraguuuBragamBradBracoonB
raciktempeBraciknyaBraceBrabunBrabbalBraapihBraaaaapppiiiiBqwBquisB
quiksilverBquicklyBquechuaBqualityqualityBqualituBqualitaeBqualiatsBquailtyBquadcoreBqtBqrainBqrBqqqBqokBqloB	qiqisachiBqingB�qgagauizkabqvaisokababaiakajbalsoksbsbwbanakoajajahabkalakahabbsksoajahbqbakaiauahgavqbakakahahabkaoaoahajakaoaohababcfahkakanbamslaljahavcxsghajnamalaihavavankakaibababmakakabbsnzkgfcnskkskabajkanabvjkanabajajjababkohaklapalnabhgvankakajajanabnzkfmfmksjsnnaksjabnakakbabbcgjaknakaljabakkanabmakajbakakabbwkaksklqnanajskkamanskldmflsowjnansmzlkxnanabjaksnsmslspksjsksknskdkjsnwnalzojsksBqeefashionshopBpxlBpwntongBpwngiriminnyaBpwlapakBpvcBpvBpuzzleBpuyengBpuyaBpuvyB!puuuuuuuuuuuuuaaaaaaaaaaaaaaaasssB.puuuuuuuuuaaaaasssssssdddddsddddddddddddddddddBpuuuuuBpuuuaaassssssssssBpuuasssB
puuaaassssBputungBputriBputivB
puterannyaBputeranBputarnyaBputarkanBpuspitaBpusanBpurgingBpurBpuoollBpuolllllllllllllllllllllllllllBpunyanaBpunyalahBpunyakBpunyaaaaaaaB	punyaaaaaBpuntirBpunggungnyaB	pundaknyaBpumaBpulungBpuluhanBpulseBpulpenxB
pulogadungBpulfenBpulasBpulanginBpulakkBpulakBpulaeBpulaaaaaaaaaaaaaaaaaaaaaaaBpulaaaB
pukulannyaB	pukalapakBpujiBpuithBpuhBpueBpuddingBpublikB
puazzzzzzzBpuazB.puasssssssssssssssssssssssssssssssssssssssssssB&puasssssssssssssssssssssssssssssssssssB#puassssssssssssssssssssssssssssssssBpuassssssssssssssBpuassssssssssssBpuasssssassaaaaaaaaaaaaaaBpuassssaBpuasssaaBpuaspuasBpuasnyaBpuaskanBpuasbonusnyaB
puasbangetBpuasatopBpuadB
puaassssssBpuaaasssssssB	puaaassssBpuaaassBpuaaasB$puaaaassssssssssssssssssssssssssssssB!puaaaaasssssssspuaaaaassssssssssdBpuaaaaasssssB
puaaaaasssB	puaaaaassB	puaaaaaasBpuaaaaaaaaassssssssBptosesBpstinyaB
psnduannyaBpsnananBpsimpldBpsheBpsesuaiBpsennanBpsckingBpsaranBpsannaaBpsananberpungsiBpruductBprubahanBprsosesB	prsanannnBprsanaannnnnB	provokasiB	protektorB	protektifBprotekBprotectornyaB
protectionB	protectedBprotectBprosuksinyaBprosotanBprosessBprosesorBprosesnyaaaBproseseBprosBproporsionalBproporsinyaBproperBpropelernyaBproofBproobleemooooB	promosiinBpromosiB	promoooooBpromblemmmmmmmmmmmmmmmmmmmmmmmmB	projektorB
projekctorB	proiesnyaBprogressiveB
programnyaBprogeamBprogamB	profilnyaBprofileBprofesionalitasnyaBprofesionalitasBprodusennyaB	produnyaaBprodukprodukBproduknyaaaB
produkkkkkB
productionB
producknyaBproduckBprodkBprocesBproceBproccesBproblmBprobleemBprobBprntingB	prnjualanBprngirimnyaBprngirimanyaBprngirimannyaBprnaB	prmbelianBprluBprlengkapanBprlapakB	prjlannyaBprjlananBprivatB
priotaskanBprioritaskanB	prioritasBprintnyaBprincessBprikitiuwwwwBpridukBprianyB	prhatikanBpreviousBprettB
pressurnyaBpressmanBpressingB
presisinyaB
presidenkuBpreodukBprengkiBpremiunBpremanBprefectBpreesureB
preeetttttB	preeettttBpredikatnyaBprdksiBprcovaanBprcayaBprbedaanBprbaiknBprankBpramuksapramuksaBpramukaB	pramugariBpramudyaBpralnBpraktiscukupBpraktikBprabowoBpqsBpqkingBpqkckingB1pppppppppppppppppppppppppppppppppppppppppppppppppB+ppppppppppppppppppppppppppppppppppppppplpppB&ppppppppppppppppppppppppppppppppppppppB!pppppppppppppppppppppppppppppppppBpppppppppppppppppppppppppBppppppppppppppppppppppppBppppppppppppppppppppBRppppppppopppppppppppppppplpppppppppppppppppppppplppppppppplpppplllllllllppppppppppBpppppppBppppppB-ppppllppppppplpppppppppllppppppplppppppppppppBppiiisssaaaannnnBpoyBpowervuB
powervideoB	powerglueB	powerfullBpowebankBpounchBpoulllllllllB
poulllllllB
potoooooooBpotonginBpotongaBpoteraBpotensiB	posturnyaBpostingannyaBpostifBpostelBpositivBpositippBpositipBpositioningBpositfBporsinyaBporinyaBporiBpopomBpopoBpoperBpooooolBpoolllBpoohB	ponselnyaBponinyaBpondanB	ponakankuBponaankuBponBpompanyaB	pomadenyaBpolyesternyaBpolybagBpolllllBpolkadotnyaBpolitikBpolisBpolibakBpokoyaBpokooknyBpokookkkknyaaaBpokonyudBpokonyabkerenBpokonyaaaaaaaaaaaaaaaaBpokonyaaaaaaaaaaaaaaaBpokonyaaaaaaaaaaaaaBpokonyaaaaaaaBpokongaBpokonayaBpokokpokoknyaB	pokokonyaBpokoknyyyaaaaaaB	pokoknyadBpokoknyaaaaaaaaaaaaaaaaaaaaaaaBpokoknyaaaaaaaaaaaaaaaaaaBpokoknyaaaaaaaaaBpokoknyaaaaaaaaBpokoknyaaaaBpokokntaB	pokoknnyaBpokoknayB
pokoknamahBpokokmyaBpokokmenBpokokknyaaaaaaaBpokokknyBpokokkknyaaaBpokokkeBpokokkBpokokepokokBpokokekBpokokeeepokokeeehhhahahahBpokokeeeeeeeeeB	pokokeeeeBpokojyaBpokoeeeeepokoeeeeeeeeBpokoeeBpokoanB
pokknyaaaaBpokknyaaBpokknyBpokkknyaBpoketBpokemonBpokeBpojoknyaBpoinnyaBpodoBpocketBpocessornyaBpobB	pntinggggBpntgBpntesB	pnjualnyeBpnjpitBpnjlsanBpnjlasannyaBpnjgBpnilaianBpnhBpngrmnnBpngrimnB	pngirumanBpngirimnnyaBpngirimanyaBpngirimanpengirimanBpngirimannyBpngirimBpngerjaanyaB	pngerjaanBpngalamnBpnduanBpndekBpnasB	pnampilanBpmisahanBpmbyranBpmbritahuanBpmbliB	pmasanganBplywoodB	plynannyaBplyannB	plyananyaBplyananBplussBplumpangnyaBplsedealBplsBplosokkcmtnBplongoB
ploknumyaaBplitB
plisketnyaB	plisketanBplinganBplihkanBplihBpletotBpletekanBpletatBplesananBplentungBplentangBplaybackBplatedBplateB
plastikpunB	plastikinB	plastikanB
plasticnyaB	plastickyBplasticBplastiBplasterBplasikB	plashdiskB	plapaknyaB	plangsingBplanganBplangBplampungBplaktisBplagBplafonBplacentaBplaBpkstikBpkrjaanB	pkonyaaaaBpkoknyaaBpkokkknyaaaaaaaBJpkoeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBpkoeBpkkonyaBpkknyaaaaaaaaaaaaaaaaaaaaaaaaaBpkirB	pketannyaBpkenyBpkengnyaBpkainyBpkaBpjBpiyamaBpixelsBpixelnyaBpiuhhhBpitikBpitaxBpissssssssssssssssandBpissBpisernyaBpisauyaBpisannnnBpisanlahBpisabBpisaanBpisaaannnnnnBpisaaannBpisaaaanBpisaaaaannnBpisaaaaaannnnnBpisaaaaaaaaaaaaaaaaaaaanBpisaBpirusBpirnyaBpiringanBpiringBpiqueBpipiBpipaBpintaBpinsilBpinnyaBpinngangnyaBpinkyzzBpinkyBpinknyaaBpinkkkBpinjamBpingpongnyaBpingirimannyaB
pingirimanBpingiinB
pinggangnyB	pindahkanBpilihinBpilihatBpilihaanBpilhBpilBpikitBpikirkuBpikirinB	pikirankuBpiketBpiiiiiisssssaaaaaaannnBpihalBpigmentBpiggangBpienyaBpieBpicsBpickupBpicknyaBpicjBpicekBpiasBpiaraanB
pianikanyaBpianikaBpiamaaaaaaaaaBpialaBpiaanBphpinB	photoshopBphotonyaBphotographedBphotochromicB	phisiknyaBphilipsBphilipBphdBphasesBphaBpgenyaBpgawaiBpfaBpezananBpeyangBpewarnaannyaB	pewarnaanBpewarnaBpewangiB	petunjukjBpetugasB	petualangBpetrickBpetotBpetnyaBpetirB
petinjunyaBpessnanBpessenBpesqnanBpesonBpesnnBpesneanBpesnannnnnnnnBpesnanaBpesimisBpesesanBpesenyBpesennnnnnnnB	pesenlagiBpeseninBpeseneBpesenannB	pesenananBpesenamBpesasanBpesansyB
pesansnnyaBpesannnnnnnnnnnnnnnnnnnBpesannannyaBpesannannnnnnBpesangB#pesanannnnnnnnnnnnnnnnnnnnnnnnnnnnnBpesanannnnnnnnnnnnnnnnnnBpesanannnnnnnnnnnnnnnBpesanannnnnnnnnnnnnnBpesanannnnnnnnnnnnnBpesanannnnnnnnnnnBpesanannnnnnnnnBpesanannnnnnnnmnBpesanannnnnnnnBpesanannnnnnnBpesanannnnnBpesanannmnnnnnnBpesanankBpesanangjufufufuguuviBpesanancumanBpesananbarangBpesanananyaBpesananannnnnnnnnnnBpesananannnnnnnnnBpesananannnB
pesanananaBpesanabBpesanaannnnnnBpesanaannnnnB
pesanaaaanBpesanaaaaannnnBpesanaaaaaaaaaaaannnnnnnnnnnnBpesamanBpesamBpesakBpesabarBpesaanaB
perusahaanBperuntukkannyaBperuntukannyaB
peruntukanB	perumahanBperubhnBpersonalB
persijanyaB	persibnyaBpersibB	persewaanBpersepsiB	persennyaB
persediaanB
persatunyaBpersatuBpersananBpersaanBpersBperpicisnyaaB
perpanjangB
perpacknyaBperosesB	pernisnyaB
pernikahanBpernasalahanBpernaBpermukaannyaB
permohonanBpermkBpermisiB	permintanBpermintaannyaB
permintaamB
permatanyaBpermasalahkanBpermasalahinB	permanentBpermanenBpermaksBpermainannyaBperlindunganBperlihatkanBperliB
perlengkapB
perlakukanBperlakB	perkuraanBperkiranB
perkirakanBperkingBperkecilBperjlnanBperjelasBperjalanannyaBperiodeBperiodBperikasaBperiaB	perhtikanBperhitungkanBperhitunganBperhhatikanB	perhatiknBperhatikannyaBperhatiannyaB	perfecttoB
perfectoooBperfeckBperezBperencanaanB	perekamanBperekamBperekaatnyaBperdanganggnyaB
perdananyaBperdaganganBpercyaBpercumahBpercsyaBpercrpatBpercmB
percetakanBperceparBpercapatB
percakapanB	perburuanB	perbuatanBperbnyakB	perbaruinBperbaruiBperbaikijahitanB
perbaharuiB	perawatanBperangkatnyaBperangBperanBperaktisB
perakperakBperakitannyaBpeptisolB	pepsodentBpepesanBpepatahBpeopleBpeokBpenyusunannyaBpenyokpenyokB	penyoknyaBpenyetrikaanBpenyetelannyaBpenyesuaianB
penyerapanB	penyerangBpenyembuhanBpenyekatBpenyekBpenyedotB
penyanggahBpenyampaiannyaBpenyambungnyaBpenyadapBpenyabarBpenyaBpenxekB	penunjangBpenumbuhBpenulisannyaBpenuhnyaB	pentunjukBpentolBpentilxB	pentilnyaB
pentilinerB	pentilasiBpentibngBpensananBpennyokB
penjulanyaBpenjulBpenjualxBpenjualnyaaB
penjualnuaBpenjualnBpenjualanyaBpenjuaaalllBpenjuaB
penjhitnyaB	penjernihBpenjelsannyaBpenjelasanyaB
penjelasabBpenjaulB
penjajakanBpenjabatBpenisBpenipuuuuuuuuuuuuuuuuuuuBpenipuannnnBpeningBpenilaiannyaBpenikmatBpenigirimannyaB
penhirkmanB	pengusahaBpengurimannyaBpenguranganBpenguncuinyaB
penguncianBpengunaannyaBpengunaaanyaB	pengumpulB
pengukuranB	penguatanBpengrmnB
pengontrolB
pengobatanB	penglarisBpenglamnBpengkuhB
pengjrimanBpengitrimanBpengitimannyaBpengirmnBpengirmannyaB
pengirkmanB
pengirjmanB
pengiritanBpengirirmanB
pengirimsnBpengirimnnyaBpengirimnannyaBpengirimmannyaBpengirimmanBpengirimbanBpengirimanyahBpengirimanterimaBpengirimanpunBpengirimannyqBpengirimannpunBpengirimannnyaB&pengirimannnnnnnnnnnnnnnnnnnnnnnnnnnnnBpengirimannnnnnBpengirimannayBpengirimancepatBpengirimananyaBpengirimamnyaB
pengirimahB
pengirimabBpengirannyaBpengiramannyaB	pengintaiBpenginstalanBpenginciB
pengiminanB
pengimanyaBpengimanBpengikutBpenghuniBpenghubungnyaB
penghubungBpengharapanBpenghapusannyaB
penghantarB	penggunaxBpenggunanyaBpenggunaanxBpenggunaannyBpenggunaaanyaBpenggulunganBpenggrimmanBpenggiringanBpenggirimanBpenggatiBpenggarisnyaB
pengganjalB
pengetesanBpengeserBpengeriminanB
pengeriminBpengerasB	pengepackB
pengendaraB
pengencangB
pengenasanBpengenaroonBpengemasannyBpengemaaannyaBpengelupasanBpengelimanyaB
pengelemenBpengelemanyaB
pengelamanBpengekB
pengecetanB
pengecasanBpengaturnyaBpengaturannnyaBpengaruhnyaB
pengarahanB
pengapuranB
pengapakanBpengantrB	pengantinBpengancingnyaB
pengamananBpengalamannyaB	pengakuanBpengaktifanyaB	pengajuanB	pengaduanBpenesBpenermaBpenerimanyaB	penerimanBpenempelB	penekananBpenegiriamanBpendistribusianBpenderitaanB	pendenganBpendekinBpendekanB
pendeeekkkBpendampinganBpencuriBpencukurBpencopetB
pencitraanBpencintaB	pencinptaBpencetannyaBpencerahanxBpencegahBpencariannyaB	pencarianBpencariBpencabutBpenataanBpenasaranyaBpenasarankuBpenasanBpenarikBpenanggananBpenandaBpenananBpenampilanyaBpenambahaanBpenakB	pemutaranBpemujaB	pemsanganB
pemrosesanBpemrogramanBpemprosesanB
pemotonganBpemisahB	pemintaanB
peminatnyaB
pemiliknyaBpemilaB	pemikiranB	pemgirmanBpemesananyaBpemesanaBpemdaBpembutanB
pembuktianBpembukaB
pembuatnyaB
pembuanganBpembomanB
pembodohanBpembersihnyaBpembersihanB	pemberianBpemberiB
pembenahanBpembelyB	pembelimuB	pembelikuBpembeliannyB
pembeliannB	pembeliamB
pembeliaanBpembatasB
pembatalanB	pembaruanB
pembalasanBpembahasannyaBpembacaB	pemasangnBpemasanganyaB
pemantauanBpemanduB	pemancingBpemanasBpemalesBpemakianBpemaketannyaB	pemaketanB
pemakaiyabBpemakaiannyBpemakaiankuB	pemakaiamB	pemahamanBpelurusB	pelumpangB
pelumasnyaBpeluangBpelopakBpelindungnyaB	peletakanBpeletBpelerBpelepakB	pelengkapBpelaynannyaBpelayannannyaBpelayananyapelayananyaBpelayanannyapelayanannyaBpelayanannyabaikBpelayanannnyaBpelayanannlBpelayananngaB
pelayananeB
pelayananaBpelayanaBpelayamamantabBpelastikBpelarianB
pelapalnyaB
pelapakpunBpelapaknyanyaB	pelapaknaBpelapaakBpelapaBpelaoakBpelanyananyaB	pelanvganBpelanggnBpelangganyaBpelangganmuBpelanggankuBpelanggangyaBpelampungnyaB	pelampungB	pelakunyaBpelakuBpelaksanaanBpelaknyaBpelaiBpelaayanannyaBpekungBpekingnyB	pekingntaBpekingnkemkurangBpekingannyaB
pekerjanyaBpekerjaanpekerjaanBpekengBpekattBpekanB
pekalonganBpekakBpejlnBpejatBpejabatB
pegununganBpeglngirimanB	pegirimanBpegiB	pegerimanBpegawaiBpegasBpeganggannyaBpeformaB	peeubahanB
peemintaanBpeeeeendeeekBpeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBpeedbackBpedasnyaBpedangBpedananB
pedagangnyBpectBpecpecahBpecokB
peckingnyaB	peckinganBpecingB	peciknganBpeciBpecahhBpecahanBpecaaahBpebisnisBpebgirimannyaBpeayanannyaBpeasanBpearingBpeakingxBpeacingBpeachnyaB	peaanannnBpeaanBpeaBpdoseaBpdlhBpdgngBpdagangBpcuknyaBpcsiBpcrBpckingxBpckgBpcingBpcarBpcahBpbxBpbrikanBpbeliBpazBpaymentBpayahlampunyaBpayahhhhhhhhhhhhhhhhhyghhhhhhhhBpayaahhhhhhhBpayaaahhBpayaaaaaaaaahhhBpaxkingBpaxingBpavingBpatokanBpatnerBpaternBpatchnyaBpatanBpatajBpatahkanBpatahhhB
patahannyaBpaswrdB
pasvdatangBpasuruanBpasukanBpastkBpastilahBpastiinB	pastannyaB9passsssssssssssssssssssssssssssssssssssssssssssssssssssssB#passsssssssssssssssssssssssssssssssBpassssssssssssssssBpassssssssssB	passsssssBpasspoBpassedBpasrahBpaspoBpasngBpasndiBpasliBpaskibraBpasingBpasinB
pasdilihatBpasdiBpasdanB	pasbangetBpasanhBpasanginBpasanagBpasanBpasalBparyasiBparselBparkaBparhumBparfumxBparebutB
parasutnyaB
parasitnyaBparaparahhhBparangBparahhhhhhhhhhBparabolaBparaahhhhhhhBparaahhhB
paraaahhhhBparaaahBparaaaahhhhhhhB	paraaaahhB	paraaaaahBpapuaBpaperB papassssssssssssipssssssssssssipBpaparkanBpanyangBpantylinernyaBpantyBpantopelBpantingBpantauanB	pantasnyaBpantapBpantangBpansBpankcingBpanjwngB	panjngnyaBpanjnginBpanjgBpanjanvB	panjangknB	panjanganB
pangkalnyaBpanggungBpanggilnBpanggangBpangetBpangeranB
pandunanyaB	panduanyaB	panduannyBpandangkacaB	pandanganBpandaiBpanckingBpancinhB	pancinganBpanceaBpanazB#panasssssssssssssssssssssssssssssssBpanasssB
panaspanasBpanaskanBpanasinBpananhyaBpananBpanahhBpampuBpalyBpalugadaBpaluB	palsuuuuuBpalsuuuuBpalsuuBpalmBpalinhBpalinganBpalikBpalibgBpaleapakBpaleBpalduB
palapaknyaBpaksuBpakrnyaBpakisBpakinyBpakinngBpakinkBpakingxBpakingnyBpakimganBpaketnyalumayancptBpaketnyBpaketinB	paketankuBpakerBpakenyBpakengBpakemBpakelahBpakeinyaBpakeingBpakeeBpakdeBpakbosBpakayBpakannyaBpakanBpakaingBpakaikanBpakaiinBpakagingnyaBpakaBpajanganpajanganB
pairingnyaBpaintBpaikBpaidBpaheBpahamiBpagimanaBpagiiiBpaggilBpaelayananyaBpaeasanBpaduanBpadsBpadhallBpadhalBpadalanBpadalahBpadakuBpadahapBpadahamBpackungBpacksBpackkngBpackkingnyaBpackkingBpackinvBpackinngnyaBpackinngBpackingyBpackingterimaBpackingrapihB	packingokB
packingnyqBpackingnyaaBpackingnB
packingingBpackingbaikBpackingbagusBpackinganyaBpackiingnyaBpackiingBpackiiinnggBpackihB	packigingBpackgingBpackcingBpackagedBpacjingBpacikingBpacibgBpachkingBpacarrBpacarpunBpacarnyaBpacarkuBpabrikannyaBpabrikanBpaatteeennnnnnBpaatiBpaasssBpaassBpaanBpaaasssssssssssssBpaaasssBpaaBozilBoyyBoyisamheheheBoxfordBownBowlnyaBovralB	overnightBoverlapBoverlaodBoverlallB
overchargeB
overallnyaBovelBovalB
outsolenyaB	outputnyaBoutnyaBouternyBouterBoutdoirBoutboundBouasBotwBotomizerBotomaticBotiginalBotherBotenBotakotaknyaBotaknyaBospekBosksksjsiwilpqjwjdBosenanBoroginalB
ornamennyaBornamenBorkessBoringBoriiiiiBoriiB	origrinalBoriginallllBoriginalitasB	originaleBoriginalcodeBorigialBorensBorederBordrrBordrerBordranBordrB!orderrrrrrrrrrrrrrrrrrrrrrrrrrranBorderrrBorderlahB	orderlagiBorderedB	orderanyaB	orderannyB	orderannnBorderaBordenBordelBordeeBordeBorchidBoranyeBorangyaBoranganBopungBoptimisBoptikBopticalBoprekBopoBoperloadBoperesiB	operatingBoperatifBoperasionalnyaBoperasionalB	operasiinB
operadikanBoperBopalnyaBopalBooppssB%oooyoyoyoyooooyoyoyoyoyoyoyoyoyoyoyoyBIoooooooooooooooooooooooooooookoooooooooooooooooookkkkkkkkkkkkkkkkkkkkkkkkB0ooooooooooooooooooooooooookkkkkkkkkkkkkkkkkkklklB#oooooooooooooooooooookkkkkkkkkkkkkkB#ooooooooooooooooooookkkkkkkkkkkkkkkB"ooooooooooooookkkkkkkkkkkkkkkkkkkkBooooooooookkkkkkkkkkeeeeeeeeeeB+ooooooooohooioooooookkkkkkkkkkjjjjjjjjjjhhhBooooooooodddddddddddddBooooooookkkkkkkkkkkkkkkkkkkkkkBooooooookkkkkBoooooookkkkkkkkkkkkBoooooooB
ooooookkkkBooooooB
oooookkkkkBoooookkkeeeBoooooB-ooookkkkkkkkkkkkkkkkkkkkkkkkkwelahhhhhhhhhhhhB-ooookkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkBoookkkkkkkkkkBoookkkkeeeeeBoookkkkBoookkkeeeeexxxxxxxxxxxxBookkkBookeBookBoohhhBooeBonyxBontingBontimeB	onlinshopBonlenyaBonlenBonkosBonkkirBongkirxBonelBondofishBonderBondeBondanganBonceBonalBomsetBomonginBomonganBommmmBommBomhBomgBomelinBomegaBomdoBombakBolongBoliveBolihBolesiBolderBolaynyBolangBolakBokwBokukeleB/oksnsjsnwuanwoqpeidbsjabaiansoabqoakzbsbsbwiwiqBoksigenB	okreeehhhBokrB$okokokpkokokokokokokokokeeeeeeeeeeeeB)okokokokookokokokokokookokokokokookokokokB&okokokokokokookokokokookokokokokokokokB0okokokokokokookokokokokokokokokokokookokokokokokBokokokokokokokookokokokokokokokBDokokokokokokokookokokkokokokkisnsbsnsksjsnsnnsnsjsksjsmsjskajdbdnskwBokokokokokokokokokookokokokokBokokokokokokokokokookB!okokokokokokokokokokookokokokokokBokokokokokokokokokokookokokokB6okokokokokokokokokokokokookokokokokokokokokokokokokokoB&okokokokokokokokokokokokokokokookokokoB6okokokokokokokokokokokokokokokomokokokokojojijomokokokB9okokokokokokokokokokokokokokokokokokokooiokokkokokooiokkoB6okokokokokokokokokokokokokokokokokokokokokokokokokokokB4okokokokokokokokokokokokokokokokokokokokokokokokokokB*okokokokokokokokokokokokokokokokokokokokokB&okokokokokokokokokokokokokokokokokokokB"okokokokokokokokokokokokokokokokokB okokokokokokokokokokokokokokokokB okokokokokokokokokokokokokokokkkBokokokokokokokokokokokokokB,okokokokokokokokokokojojokokokokokokokokokokBokokokokokokokokokkB!okokokokokokokokkkoooooookkkkkkkkB!okokokokokokokokkkokokokkkokokokkBokokokokokokokokijikBokokokokokokokokBokokokokokokokB	okokokokoB	okokokokkBokokokokikokokokimkkokokokokokBokokoBokokkkkkkkkkkkkkkkkkkkkkBokokkkBokokkBoknumBoknamunBokmantapBoklkkkkkkkkkkkkkkkkkkkkkkkkkkkkBoklahhhBoklaahBoklBokkkmantaappplBokkkkkrreeeeeeeeeeB"okkkkkkkkkokkkkkokkkkkokokokokokokBokkkkkkkkkkkokkkkkkkkkkkkkkkkB$okkkkkkkkkkkkkkkkkkkkkkkokkkkkkkkkkkB"okkkkkkkkkkkkkkkkkkkkkkkklkkkkkkkkB'okkkkkkkkkkkkkkkkkkkkkkkkkkkkokokokokokB)okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkB(okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkB'okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkB&okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkB%okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkB#okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkB"okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkBokkkkkkkkkkkkkkkkkkkkkkkkkkkkkBokkkkkkkkkkkkkkkkkkkkkkkkkkkkBokkkkkkkkkkkkkkkkkkkkkkkBokkkkkkkkkkkkkkkkkkkkkkBokkkkkkkkkkkkkkkkkkBokkkkkkkkkkkkkkBokkkkkkkkkkB
okkkkkkkkkB	okkkkkkkkBokkkkkBokkeyBokkeeBokiBokhargaBokeyyyyyyyyyyBokeyyyBokeyyBokewBokeuBokesipppppppppBokeokeokeokeokeokeB!okeokeokeokeoekeoekeoekeoekeekekeBokeokegagahahahahananananananaB-okeoekeoekepelepeoejsnsbbsnsnnsnsnnsnsnsnsjssB okeoekeoekeokeoeksksossnkekekekeBokeocwoceoceoceoceoceBokentokonyaB
okelahjjhjBokelahhhhhhhhhhhhhhhhhhhhBokelahhhBokelaahhhhhhhhhbbbhbbhhhhhhhhjjBokelaaaaahhhhhBokelaBokekekekekeB8okehokehokeokeokehokehokeokeokehokehokeokeokehokehokeokeBokehokehBrokehhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhnhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhBokehhhhhhhhhaB
okehhhhhhhBokehhhBokeexxxxkkkkkkkkkkkkkkkkkkssssBokeehBokeeehhBokeeehBokeeeerBokeeeellllaaaahhhB
okeeeeeeewBokeeeeeeereeeeeeeeeBokeeeeeeelahhhhhhhBokeeeeeeeeeeewB&okeeeeeeeeeeeeeeeeeokeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeeeeerB+okeeeeeeeeeeeeeeeeeeeeeeeeeeeseeeeeeeseesseB=okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeerreeeeeBAokeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeB:okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeB7okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeB6okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeB2okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeB*okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeB(okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeB okeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeeeeeeeBokeeeeeeeeeeeeB
okedahhhhhBokealahBokbrgBokbgsB	okayyyyyyBokayyyBokaayyyBojvBojolBojoBohmBohhhBohhBohenBogahanB	ogaaaahhhBofferallBoferBoetunjukBoesenanBoesenBoesanB
oermintaanB
oerminraanBoercepatBoentingBoengirimannyaBoengelihatanBoekelahBoeeBodolBoderanBoddBocheBoceoceBoceeeeeeetttttttBoceanBobrolanBobrasanBobralanBoblongBobjektifBoberallBobengnyaBobaBobBoatchocoBoastiBoangkirBoalahBoalaahBoakingBoakeBoakaiBoadaBoackingBnzooajbsimakxbxdjixnjxBnyyBnyusutBnyusunBnyusulnyusulBnyusuiBnyustBnyusBnyukurBnytanyaBnysellBnyqmanBnyosBnyoplokBnyopirBnyontoBnyongeBnyongBnyolokinBnyolokBnyobakBnyobaanBnyoakBnymprBnylonBnylaBnyindirBnyincingBnyimengBnyieunBnyettingBnyetritB	nyetelnyaBnyesuainBnyessalBnyeseulBnyesellllllllllB
nyeseeelllBnyervisBnyerahBnyeplosBnyentuhBnyentrikB	nyemprengBnyempilBnyelipBnyelelBnyekerBnyekelBnyebutinBnyebarBnyatokBnyatamyabukuranBnyarinyaBnyariinBnyapuBnyapanB	nyanyinyaBnyanyiBnyanyaaaaaaaaaaaaBnyanyB	nyantolinBnyantolBnyanpekBnyankutBnyangkulBnyangBnyandetBnyananBnyamvkBnyamprkBnyampeyBnyampeknyaaBnyampekkBnyampekeB
nyampeinyaBnyampeinBnyampeeeeeeeeeeeenyaBnyampeeBnyampanBnyampaixB
nyampainyoBnyambutBnyambungnyaaaBnyamapiB	nyamannyaBnyamannnnnnnnnnnBnyamannnB	nyamanlahBnyamandiBnyamananBnyamanaBnyamaiBnyamaannBnyamaanBnyamaaaaaaaaaaaaaaaaaaanBnyalonnyalonBnyalapunBnyalaaaaaaaaaaaaaaaBnyalaaBnyalBnyakkBnyakarBnyahhhhhhhhhhhhhhhBnyahhhhhBnyagkutBnyadikurangiBnyadarBnyacumnBnyablonBnyabagusBnyabBnyaaaaaaaaaaaaaaaaaaaaaBnyaaaaaaaaaaaaaaaaBnyaaaaaBnxixjsBnxBnvchBnuzieBnuwuuunBnuwonBnuviBnutupnyaB	nusawunguB
nusagrosirBnusaBnurutBnuruninBnurshopBnursalamBnurraniBnuragBnurBnunguinBnundukBnumpangBnukerBnukarBnuhuhnB	nuggetnyaB	nugetnyaaB	nubagusnaBnuajBnuaBntuBntonBntnBntkBntinyaBntfsBnteB0ntapssssssssssssssssssssssssssssssssssssssssssssBntapssssssssssssssssssBntapssssssssssBntapsssBntapssBntapppsB*ntapppppppppppppppppppppppppppppppppppppppBntapppBntappBntapntapntapntapntapBntaozBntahlahB&ntabhhhhhhhhhhhxhsjsbshshshdbdbdhesvvdB
ntabbzzzzzBntaahB0ntaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbBntaaaapBntaaaaappppBntaaaaaaaaaaaaaaaaappssBntaBntBnsvhdjhdBnsnsnsnsjsjskaiaoaoaoaoskB�nsjswgsyeuehshsabsnjsduhddujsjsjsjeusbshshhejsjsnsnsjsjeueujjwjsjskaksikswiwjwiwjejsnsnnsjskejeejsjskwieieeiejsjejejeisoskarntgdyyduejeksndfiriejdjBnrmalBnrimaBnrawangBnrBnpBnoyBnoxBnoviBnovemberBnovaBnotipBnotifikationBnotifikasinyaBnoticeB	notbadnotBnotbadBnotanyaBnotabeneBnostrudB	nostalgicB
normaaaallBnorBnooooooooooooooooooooooooooooooBnooooooooooooooooBnoooBnonyaBnonyBnonusnyaB	nontonnyaB
nonggoooolBnoneBnomrBnomplokBnominusBnomermyaBnomberBnoisenyaBnoiceBnohBnockB	nobgkrongBnnxnnnnnmmmmBnntinyaB#nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnBnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnBnnnnnnnnnnnnnBnnnnnnnnnnnnBnnnnnnnnBnnnnnnnBnnnnB+nnnmmnnnnnnnkhgttdrrdddcfttffffffvggghghhgfBnnnmmmmBnnnBnmrnyBnmpelBnmnmnmnmnmnmnmnmnmnmnmnmmnmnmBnmmdndjBnmmajjssjsbiqjaBnmerBnmcmnBnmbernyaBnmanyaBnlfnBnlBnkjB3nkhznzbsjsksnksjsbsksbsnshsbsnsbsbshshshsjsnsnssnsjBnjyyBnjosssBnjirBnjgyBnjckmmlBnjalukBniyBnivoBnittakuBnitnotBnitendoBnitakuBnisiBnirunyaBnirmalBnirkabelBniqBnipplesBningratBningguB	ninggalinBnimirBnimbulBnilonBnilepBnilaBnilBnikmatttBnikmatinBnikmatiBnikiBnikayuBnikahanBniiiyBniiiiiicccceeeeeeeeeBniiceBnihkzlBnihhhhhhhhhhhhBnigthBniehBnieehhBnickBnichhB,niceniceniceniceniceniceniceniceniceniceniceB nicegoodnicegoodnicegoodnicegoodB0niceeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeB(niceeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBniceeeeeeeeeeeeeeeeeeeBniceeeeeeeeeeeeeBniceeeeeeeeeeeeB
niceeeeeeeBniceeeeBniceeeBniannnnnBniaBnhvgBnhgBnhexBngusapxBngusBngurusinBngurusBngurangiBngunciB
ngunainnyaB	ngumpetinBngumpetBngumetBngulurBnguletBngukurBnguentenBngucurBngucekBngubahBngshBngrimnyaB
ngrakitnyaBngrakitBngperesBngpasBngotakBngorekinBngopyBngopilahBngontrolBngontrexBngonekinnyaB
ngomongnyaB	ngomonginBngomngBngobrolBngoBngmongBnglupassssssssBnglupasnglupasBngliatBnglemparnyaBngjrengBngjaminBngirimnyB	ngingetinBngineoBngincerBngiluBngilerBngiketBngibulBnggungBnggihBnggggyyyyyyytddfffBngggaBnggayaBnggaweBnggausahBnggapBnggalBngfreshBngflashBngeyelBngeuriBngeunahBngetttttBngetrekBngetilepBngetepelBngeshotBngesetBngerusakBngerosokBngerjainBngeriputB
ngerespondB	ngeresponBngeresapBngerekamnyaB	ngerecordBngerdropB	ngerayainB	ngerapiinB	ngerantauBngerakitnyaBngerakitBngeprintB	ngeprezzzB!ngepresssssssssssssssssssssssssssB7ngeplaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaakB	ngeperessBngepelB	ngepassssBngepasngepasB
ngepacknyaBngepackBngengganBngeneessBngendonBngendapB
ngemodelinBngembungBngembunBngembalikinBngemaluBngelossBngelolohBngelokBngelirikBngelilitBngelihatBngelesBngelepasB
ngelebatinBngelapakB	ngelanturBngelanjutinyaB	ngelanganBngelacakBngekliknyangekliknyaBngeklikBngejamBngehBngegymB	ngegulungBngegripBngeglasBngegassB	ngeganjelBngegamesBngefresBngeflashBngedroopBngedofB
ngecewaiinBngecewaBngecengB	ngeceknyaBngeceBngecastBngecasssBngecargeBngecaasBngebungkusnyaB	ngebulmahBngebulllllllllllllBngebulbulbulbulB
ngebuktiinBngebrikB
ngeboseninBngeboongB	ngebohongBngebingunginBngebetBngebasssB	ngebasnyaB
ngebalikinBngebalesnyaBngebalesBngebacotB
ngebacanyaBngebacaBngdropBngcasBngbntukBngbassBngbasBngayalBngawasiB	ngaturnyaBngatasinBngataBngasinyaBngasiBngashBngarusB	ngaruhnyaB	ngaretnyaB	ngarepnyaBngarepinB	ngareepppBngapurBngapretBngappaBngapapaBngapaB+nganunganunganunganunganunganunganunganuasuBngantungBngantukBngantriB
nganternyaBnganterinnyaB
ngantarnyaBnganjukB	nganggonaBnganggoB	ngangetinB	ngangeninBngancingBngamukBngampusBngamenBngambungBngaleminBngalahinBngakuuuBngakkBngakalinnyaB
ngakakkkkkBngakakBngajarBngadepinBngacirrrrrrrrrrrBngaciiirBngaciiiirrrBngacengB
ngabuburitBngabrinBngabarinB@nfsngdkgdkhxmfsitsheueludktelufgaurwkydmgsitekhskydlixmhdtidkhdiBnfsBnfdBnfcBnexxxxBnexttimeBnextakuBnexrBnewbieBnewbeBnevyBneverBnetworkBnetizenBnetgearBneterBnetcellindoBnetbookBnestBnesinnyaBnerkBnerjalanBnerimoBnerimnyaBnergenBnerekkkkB	nerawanggBneonBneoBnentiBnentengBnencerminkanBnenarikkBnemuinBnemenihBnemenBnembusBnembeBnelpunBnelpBnelfonBneleponBnelayaniBnekatBnekadBnehhBnegonyaB	neglasariBnegeiB
negatifnyaBnegatifeBnegaraBneedBnecinyaBnechBnebakBndutB"ndjdkdkdkddksksldldkkddkdkdkhdhdjdBndisekBndesoBnddjuehedbdjrnfdirnrjdjdjhdbdBndavyBndapapaB
ndaaaaaaanBncurBncBnbcB	naympenyaBnayamanBnayaBnawarinBnavynyaBnavvyB
navigationBnatuteBnatreBnasionalnyaBnasiibBnasBnaruhnyaBnaruBnarsisBnarngBnarayaBnanyakBnanyainBnanunBnantikanBnantapBnantabBnanjakBnangkpBnangisB4nananznznnnznnznznzbznznnznzbbzbsbnbssnsnsnsjsjsksksBnanangBnanananaBnamuBnampungBnampangBnamgkapBnamesetBnamesateBnamayaBnamaunBnamanyaaBnamanyBnakutBnakkBnakedB(nakakbabwnsjdodormrnkeksnsbabakakbakwknsBnajisBnaiknyaBnahusB
nahhhhhhhhBnahhBnagkepB+nagabamajagavajakanacakaksoaksksbsgvskslalaBnafsuBnafkahBnafasBnabungBnaaaBmzksBmzBmywantapppppBmyusulBmyuaanttapppBmykaBmybamusBmyaaB
mwulusssssB
mwncobamyaB
mwemuaskanB4muuuuuuaaaaaaaaannnnnntttttttaaaaaaaaaaaapppppppppppBmuuulluussssBmuurahBmuuaanntaapppBmuuaachhB
mutusesuaiBmuterinBmutenyaBmuteBmutahbhrsnyaBmutahBmutBmustinyaB
mustardnyaBmustardB	muslimnyaBmusliminBmuslimahBmuskBmusicboxnyaB	musicallyBmurudulBmursidahBmurshBmurnyaB
murmerrrrrB	murmermewB	murmerlahB
murmercengBmuridBmuriahBmurhaBmurceeBmurahxBmurahsnBmurahlahBmurahhhhhhmurahhhhhhhhhBmurahhhhhhhhhhhhhhhhjhhjhhBmurahhhhhhhhB
murahhhhhhBmuraheeeeeeeeBmurahannnnnBmurahanmurahaaaaaaaaaaanBmurahaaaannnnnnnBmuraganBmuragBmurabB	muraaahanB
muraaaahhhB	muraaaaahB
muraaaaaahB"muraaaaaaahhhhhhhhhhhhhhhhhhhhhhhhBmuraaaaaaaaaaahhhhhhBmuraBmuofeatB
munyilllllBmunyakBmuntahanBmuntahBmunkinBmungutBmungknBmungkknBmungkirBmunggilBmuncakB
munafienyaBmunafiBmumusimgBmumpuniBmumetBmumerrBmuluzzzB	mulusssssBmulussBmulusnyaBmultitesterBmultitabB
multimediaBmultifunctionBmuliaBmulanyaBmulailahBmukrnaBmukminatB
mukenahnyaBmukeBmukaanBmujurBmujarapBmuirahBmuhamadBmugiBmudiBmudhnBmudhanBmudahmudahanBmudahahBmudagBmubgkinBmubasirBmubadzirBmuayanB	muattttttBmuatttB	muatannyaBmuaskanBmuasinBmuarahBmuaraBmuanteppppppppppBmuanteeppssBmuantappppppBmuantappBmuantaoBmuantabbbbbbsssB	muantabbbB
muantaafffBmuantaafBmuantaabBmuantaaapppB
muantaaaffBmuantaaaapppBmuantaaaabhBmuantaaaaaaapppBmuantaaaaaaaaaaaaapBmuantaaaaaaaaaBmuannttaapppBmuakBmuaantaaappBmuaaatB
muaaaraaahBmuaaaaannnnttttttaaaapppBmuaaaaaaaaaaaaatBmttbbzzBmtpBmtiinBmsxBmsuBmsskipunBmssihBmslkanBmskshBmskpnBmskhBmskasihBmskanBmsizeBmrudulBmrsaBmrnurutBmrngirimBmrlarBmrkaBmripBmrahBmqnqBmpsBmpoBmoyongBmovingBmouspadBmountingBmotufBmottonyaBmottoBmotornyaBmotonyaBmotionBmotifnyBmotherboardBmostBmoselnyaBmorningB	morfologiBmorensBmonthBmonopolyBmonopolinyaB
monohidratBmondarBmoncrotBmonaBmomentBmomenBmoltoBmoltenBmolornyaBmoldingBmokutonBmokkaBmokatBmojokBmoisturizingBmoggaBmogamogaBmogaemuaskanBmogBmodwlnyaBmodissBmodifanBmodemnyaBmodelnaBmoccanyaBmobilnyB	mobilitasBmobilanBmobaBmoantepBBmoannnnnnmmnmmnnntttttttappppppppplpppppppppppppppppppppppppppppppBmnyesalB
mnyebalkanB	mnyatakanB
mnyamarkanBmnyakBmnumBmnuaskanBmntppBmntpjjgvhgvfgjkknbfgtvgkkjbbvgBmntiBmntepBmnteeeeeeeeeeepBmntbbbBmntavBmntapppppppppBmntappppppppBmntapppppllllllBmntapppppbakaB
mntaomntaoBmntaffBmntabbbbbbbbbbBmntaapBmntaaapBmntaaaappppBmnnceBmnjurBmnjualB	mnjelekanBmnjdiBmnimalBmniBmngurusBmngkilapBmngisiB	mngirimknB
mngirimkanBmngiknB	mngetahuiB
mngesankanBmngembalikanBmngeB	mngcwaknnBmngatasnamaBmngambilBmnfaatBmnegecewakanBmndratBmndengarBmndaratBmndapaknBmndadakBmnchargeBmncaappBmnarikB	mnanggapiB
mnambahkanBmnahanBmnagpangonaBmmuaskanBmmpengaruhiBmmmuuuuaaaannnntttaaappppBmmmnBmmmmmmooooooooonnnnnnnB*mmmmmmmmmmmnmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmBmmmmmmmmmmmmmmmmmmmmmmmB2mmmmmmmmmmmmmaaaaaaaaaaaaannnnnnnttttttaaaaaapppppB>mmmmmmmmmmmaaaaaaaannnnnnnnnntttttttttttttaaaaaaaaapppppppppppB
mmmmmmmmmmB)mmmmmmmmmaaaaaaaaaannnnnnnnntttttaaapplllB4mmmmmmmmaaaaaaaaaaaannnnnnnnnntttaaaaaaaapppppppppppBmmmmmmmmB*mmmmmmmaaaaaaakkkkkkkkaaasiiiiiiiiiiiiiiiiB"mmmmmmaaaaannnnntttttaaaaaapppppppB?mmmmmmaaaaaaaaaannnnnnnnnnnttttttttttaaaaaasssaaaaaaaapppppppppBmmmmmaaannntttaaaplahB#mmmmmaaaaaannnnnnnttttttaaaaappppppBmmmmmBmmmmaaaakkkkaaaasssiiihhBmmmmaaaaaayyyyyyaaaaannnnnBmmmmaaaaaannntttaaaapB(mmmhshshshshshshshshshshshshshshshshshshBmmjsjkkskskskaamBmmgfhiooouhfghujijBmmbthknBmmbersihkanB	mmberikanBmmbeliBmmbacaBmmangBmmahBmmaanntttaappppBmmaannttaaabbbBmluBmlimpahBmlhtBmlgBmleyotBmletoyBmletoiiBmlesetBmlencengBmlahanBmkwoaksBmksudBmkskBmksjB	mksiiiihhBmksiihBmksiiBmksihhhhhhhhhhhhhhhhhbhhBmksihhhhhhhBmksihhhhBmksihhhBmksihhBmkshhhhhBmkshhhhBmksdBmknyBmknanBmklumBmkeBmkdhBmkchBmkassiihhhhhhhhhhhhhhhhhBmkasihhhhhhBmkasihhhBmkashnyaBmkanBmkainyaBmkaiBmkaasihBmkaaihBmkBmjuBmjdBmjBmitreBmitosBmiterBmitchBmitaBmisuaBmistyBmistikBmistB	missrouteBmisscomunicationB
misleadingBmisalpunBmisalkanBmisahBmisaBmisB	mirrornyaBmirroringnyaBmiripanBmiringinBminyaknyBminutesB
minumannyaBmintanBmintainBmintaiBmintaaBminoxnyaBminoxidilnyaBminoxBminnyaBminnowB
minimarketBminhumBminguuBminggirBminatiBminannasB
minannaaarBminahBmimitiBmiminnBmimaxBmilonyaB
millenialsBmiliterB	milimeterBmiliBmildBmikroponBmikirinBmikatBmiiinBmiguelBmigaBmifiBmidleBmidahBmicxB	microsoftBmicrophonenyB	microfoneBmicrofonB
microfiberBmicinBmicgeekBmibandBmhunBmhkshBmhhBmhantabzBmhantablaahBmhantaapBmhalBmggBmgedropB
mgecewakanBmgcwaknBmeyesalB	meyerupaiB
mewakilkanB
mewajibkanBmeujeuhBmetroxBmetroBmetongB	metodenyaBmethodBmetekB
metaliknyaBmetBmesuaiBmestinuaBmestakBmessagesBmessBmesmewB
mesmesdariBmeskipuBmeskipnBmeskipinBmeskiiiBmesinyaB
mesenmesenBmeseluruhanBmesekBmerugiBmersaB+merrrdeeeeekkaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBmerkmyaBmeriyahBmeriksaBmeriahhhhhhB
meriahhhhhBmeriahhhBmeriahhBmeriaahhhhhhhhB
meriaaaaahBmeriaBmeriBmereturBmerethurB	merespontB
meresponseB	meresponsBmeresapiBmerepresentasikanBmerengekBmerengBmerenBmeremBmerelaksasiBmerekmyaBmeregangBmerefundBmereferensikanBmerecordBmerecomendedBmerduB3merdekaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBmerckB	merayakanB	merapikanB	merahnnyaBmerahhBmerabaBmerB
meperbesarBmenyusutBmenyuapakanBmenyolokBmenyitaB
menyingkirB
menyimpainB	menyikapiB	menyetingBmenyetelnyaBmenyesuiakanBmenyesatkannyaBmenyesalllllB	menyesallBmenyerapnyaB	menyerangB	menyerahhB	menyentuhBmenyembunyikanB
menyejukanBmenyeimbangkanBmenyehatkanBmenyebarBmenyebalkanBmenyebabkanBmenyayangkanB
menyatakanBmenyarankanB	menyangkaB	menyanggaBmenyandingkanBmenyampaikanB
menyamakanBmenyamaiBmenyalakannyaB
menyalakanB	menyakitiBmenyadapB	menutrisiBmenunyaBmenunjukB	menunjangBmenunggunyaBmenuneB
menulisnyaB
menujukkanBmenueutBmenuduhBmenualBmentongiBmentongBmentolBmentionBmentholBmentariBmentahBmensongBmenshootBmensBmenonaktifkanBmenolakBmenntingBmenjwlaskanBmenjwbBmenjeritBmenjelekkanB
menjelekanBmenjawabnyaB
menjajikanBmenjahitnyaBmenjadikannyaBmenirimaBmeninggalkanBmeningB	menikmatiBmeniB
mengutmkanB	mengundurBmengunakannyaBmengulurBmengujiBmengugahBmengucapkanB
mengubungiBmengoperasikanBmengonsumsinyaBmengolorB	mengkotakBmengkoreksinyaBmengkonversiBmengkonfirmasikanBmengkoneksikanBmengkoleksiBmengkoBmengklikBmengkilapkanBmengkhawatirkanBmengkecewakanBmengisiB	mengirimnBmenginsfirasiB	menginjakBmenginginkannyaBmengingatkanBmenginformasikanBmengikisBmengikatBmenghubungkanBmenghubunginB
menghubugiB	menghitamBmenghiraukanBmenghindariB	menghijauBmenghidupkanyaBmenghianatiBmenghendakiB	menghematBmengharapkanB	mengharapBmenghancurkanBmenghalalkanB
menghadiriB	menggunknB
menggunkanBmenggunakanyaBmenggunakanxB
menggulungBmengguB
menggrusukBmenggoncangBmenggodaBmenggembungBmenggembirakanBmenggemaskanB
menggeliatBmenggelegarrrrBmenggelegaaarrrB	menggarukBmenggantungBmenggantinyaBmenggambarknBmenggaliB
mengewakanBmengetiknyaBmengerriBmengerjakanBmengepaknyaBmengendaraiBmengenalkanBmengenakkanB
mengenakanBmengenaB	mengempisBmengeluhBmengelakB
mengelabuiBmengejarB	mengecwknBmengecohBmengecikBmengecewakanmengecewakanBmengecewakanbarabgBmengecewakamBmengecewakaB
mengecewakBmengeceakanBmengeceBmengecawaknB	mengcoverBmengcopyB
mengcewaknB	mengawasiBmengatasinyaBmengasikkanBmengarahkanBmengarahBmengaplikasikannyaBmengaplikasikanBmengapBmengantianyaBmengantiB
menganggapBmengangaBmengambilnyaB
mengambangB	mengamatiBmengalirkanBmengalahkanBmengakibatkanBmengajiB
mengadakanBmengabadikanBmengaB	menetukanBmenetesBmenerimanyasesuaiBmenerimanyaBmenerbangkannyaB
menerapkanBmenerangkanB	menerangiBmenepiBmenentuB	menemukanBmenempatkanBmenembakBmenelanBmenekanBmenecewakanBmenecekB
menebalkanBmenebalBmendungBmendugaB	mendorongBmendidikB
mendeteksiBmendesakB
mendengungBmendengerkanB	mendemmmmBmendelayBmendelapBmendekatBmendatangkanB
mendatangiBmendarattttttttttttttttttttBmendamBmendalamB	mendaaratBmencuriB	mencretttBmencolokBmenciutB
mencewakanBmenceritakanBmencepitB
mencederaiB
mencarinyaB
mencarikanB	mencampurBmenbawaBmenawanBmenasrikBmenaruhB)menarikkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkBmenarikkBmenangissssB	menanganiB
menanamnyaB
menampakanBmenambahkanB	menamakanBmenakutiBmenakjubkannnnnB	menaikkanBmenaeikBmenabungBmemyampaikanBmemuuuuaskanB
memuuaskanB
memutuskanB
memutarnyaBmemuskanBmemungkinkanBmemulaiB	memudahanBmemuasssskannnnnnnnnnnnnnnB4memuaskannnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnBmemuaskannnnnB memuaskanbsjsksksksllssmsmsmsmmsBmemuaskanbarangnyBmemuaskaaannnnBmemuaskaaanBmemuaskaB	memuadkanB.memuaassssskaaaaaannnnnnnnnnnnnnnnnnnnnnnnnnnnB	memuaakanBmemuaaassskkaaaaannnBmempunyaBmempuniBmemproduksiBmemprihatinkanBmemprhatikanBmempersiapkanBmemperpanjangBmemperlihatkanBmempercantikBmemperbaikinyaBmempelajarinyaBmemotretBmemorryB
memoricardBmemlaluiBmeminimalkanBmeminimalisirBmemilikB	memilihknB	memghiburBmemgecewakanBmemfungsikannyaB
memerlukanBmemerBmemengaruhiBmemeberiBmemdingBmembuntuBmembuktikannyaB
membuatnyaBmembuangBmembosankanB	membludakBmembleB
membimbingBmembicarakannyaBmembesarmanBmemberkatimuBmemberitahuBmemberatkanBmemberB	membentukBmembekuBmembdohBmembayarnyaB	membayangBmembaraBmembantuuuuuuuuuuuuuuuBmembalaskanBmematahBmemasukiBmemasangnyaBmemasangkankanBmemantulkanB	memanjangBmemanduBmemamgB
memakannyaB	memainkanB
meluruskanB	melungkerBmeluberBmelorootBmelodyBmelisaBmelintirB	melintangB	melingkarBmelimpahBmelilitB
melihatnyaBmeliBmeletotB
melesatnyaBmelepasB
melengkapiBmelemparB	melelelehBmelekB	melegendaBmeledukB
melebihkanBmeleberBmelayangB	melayanaiBmelarnyaBmelapisiB	melangganB	melampauiB	melakukabBmelakB	meinstrimBmeilibahenlingnyaBmehonggggggggggBmegeBmegabassBmegaaccBmeetingBmeeeehBmediasiBmedaratBmecoolBmecolokB	mecinglahBmecetotBmeccaBmecariBmeakiBmdchBmdaratBmdahanBmcmBmcgBmcetBmcepeatBmcBmbulakBmbulahB	mbukaknyaBmbukaBmbuhnpeBmbuhnBmboisBmblB
mbingunginBmberudulBmberiknBmbekasBmbatBmbaeBmazBmayuraBmayungBmayitB	maybelineBmayanlaaaaahBmayaannnnnlahhhBmayaanB	mayaaannnBmayaaanBmayaaaanB	mayaaaaanBmayaaaaaaannnBmaximumBmaxiBmawBmavBmauuuuuuuuuuuuuuuuBmauuBmauraBmaungkinBmaumauB	maubreturBmauanB	maturswunB+maturnuwuuuuuunnnnnnnnnnnnnnnnnnnnnnnnnnnmmBmaturnuwhunBmatricB	matrasnyaBmatinyaBmatinyBmatiknBmatiiB	materinyaB
materialnyBmatengBmatebBmateBmatcingBmatchlahBmatchaBmatapssssssssssssssssssBmataplahBmatangBmatakuBmatahB8matabsssssssssssssssssssssssssssssssssssssssssssssssssssB(matabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbvbbbBmataaaaaaaaappBmasyaaBmasuuuuuuuuukBmasuuukBmasukxB	masuknyaaB
masukannyaBmasterpieceB)masssssssssntaaaaaaaaaaapssssssssssssssssBmasssageBmassageBmassaBmassBmasooookkkkBmasokBmasnyaBmaslhB
maslaahnyaBmaskeryaBmaskernyBmasjidBmasimusBmasihhBmasigBmasibB
mashaallahBmasdaBmasanginB
masalahkanBmasalahhBmasaknyaBmarunnyaB	marronnyaBmarronBmarrkottoopBmaroonxB	maroonnyaBmarmerBmarlboroBmarkotosBmarkotoppppppppppppppppppppBmarkotopppppBmarkotoppppB
markotopppBmarkotoooopppBmarkotoooopBmarkotoooooopBmarkotooooooopppppBmarkotooooooopB)markotoooooooooooooooppooooooppppppppppppB9markotoooooooooooooooooooooooooooooooooooooooooooooooooopB	markotobzBmarkotobBmarkojooosssBmarkitopBmarkisipBmarkingB
markicokssBmarketoooooooooopB	marketingBmarkatopBmarinaBmargotopB'margonjooooooooooooooooooooooooooosssssBmaradonaBmarBmapnyaBmaoannavBmanyaBmanyBmanusiaBmanulBmanufieBmantuuuuuuulB
mantuuuuulBmantulllllllllllllllllllllB
mantuaapppBmantuBmantttttaappppppppBmanttttaaaapppppppppppppppppBmantttaaaappppppppBmantttaaaaapB	manttavvvBmanttapppppppBmanttapBmanttaaaappppB	mantreeepB
mantrappppB
mantraapppB
mantraaaapBmantppppppppppppppBmantppppBmantplahBmanthafBmanthabbbbbB	manthaaapBmanteupppppppBmanteupB
mantepzzzzBmantepssBmantepsB)manteppppppppppppppppppppppppppppppppppppB mantepppppppppppppppppppppppppppBmanteppppppppppppB
mantepppppB
manteppppoBmantepmantepmantepmantepBmantengBmantelBmanteeppppppppBmanteepB	manteeeppB+manteeeepppppppppppppppppppppppppppppppllllB!manteeeeeppppppppppppppppppppppppB
manteeeeeeBmanteeeeebbbbbbbBmantebsBmantbBmantavvvvvvvvvvvvvvvB	mantatabsBmantatBmantarBmantapzzzzzzzzzzzzzzzzzzzzzzzzB
mantapzzzzBmantapvvvttB	mantaptapB!mantapsssssssssssssssssssssssssssBmantapssssssssssssssssssBmantapsssssB	mantappssBmantapppsihsesuaipesananBmantappppsssssBmantappppsssBmantapppppssssssdBmantapppppppppppssssssBWmantappppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppBKmantappppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppB/mantappppppppppppppppppppppppppppppppppppppppppB/mantappppppppppppppppppppppppppppppppppppppooooB+mantappppppppppppppppppppppppppppppppppppppB(mantapppppppppppppppppppppppppppppppppppB'mantappppppppppppppppppppppppppppppppppB&mantapppppppppppppppppppppppppppppppppB$mantapppppppppppppppppppppppppppppppB#mantappppppppppppppppppppppppppppppB$mantapppppppppppppppppppppppppppplppB'mantappppppppppppppppppppppppppppllppppB!mantappppppppppppppppppppppppppppBmantappppppppppppppppppppppBmantapppppppppppppppppppppBmantappppppppppppppppppppB!mantapppppppppppoppopoppoppppppppB"mantapppppppppppooopoppoooppppppppB!mantappppppppppmantapmantapmantapBmantapppppppppmantapppppppppggBmantapppppppplmantappppppppBmantapppppppopooooBmantapppppppllB*mantappppppopppppppppppppppppllpppppppppppBmantappppppolllllllBmantappppppoBmantappppplBmantapppplllllB$mantapppmantapppppmantapppmantapppppBmantappplahB	mantapplpBmantapplllllB
mantapplahBmantapokokelahB$mantapmantapmantapmantapmantapmantapBmantapllBmantaplahhhBmantaplBmantapkiBmantaphBmantapbsssssBmantapbBmantaopmanthappBmantandBmantafsB	mantaflahB1mantaffffffffffffffffffffffffffffffffffffffffffffB'mantaffffffffffffffffffffffffffffffffffBmantaffffffB
mantabzzzzB	mantabzzzBmantabzzBmantabsssssssssssssssBmantabsssssB
mantabssssBmantabsbarangBmantabpsBmantabmantabB
mantabjiwaB	mantabhhhBmantabhBmantabekBmantabeeeeeeeeeBmantabdB
mantabbsssB	mantabbssBmantabbbzzzzBmantabbbsssssssB#mantabbbbbbbbbbbbbbbbbbjjjjjjjjjjjjB@mantabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbB*mantabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbBmantabbbbbbbbbbbbbbbbbbbbbbBmantabbbbbbbbbbbbbbbbbbBmantabbbbbbbbbbbbbbbbbBmantabbbbbbbbbbbbbbbbBmantabbbbbbbbbbbbBmantabbbbbbbbbbbB
mantaapsssBmantaapsB/mantaapppppppppppppppppppppppppppppppppppppppppBOmantaappppppppppppplplpppppppppppppppppppppppppppppppppppppppppppppppppppppppppBmantaapppppppBmantaappppppBmantaappoooBmantaappfpppppBmantaaffffffB
mantaabzzzB
mantaabbzzBmantaabbblahBmantaabbbbbnnnBmantaabbbbbbbbB
mantaabbbbBmantaaapsssB!mantaaappppppppppppppppppppppppppBmantaaapppppppppppppBmantaaapppppppppppBmantaaapppppppppBmantaaappppppppB%mantaaappppapapappapapappapapapappapaBmantaaappplBmantaaamantaaaapBmantaaafBmantaaabsssBmantaaabplahB%mantaaabbbbbbbbbbbbbbbbssssssssssssssBmantaaabbbbbbbbB
mantaaabbbB	mantaaabbB
mantaaaaspB
mantaaaapsBmantaaaappppppppppBmantaaaapppppppppBmantaaaappppppppB
mantaaaappBmantaaaapmantaaaapBmantaaaaplahBmantaaaaoppppppBmantaaaafffB
mantaaaabsBmantaaaabbbbbbBmantaaaabbbbbBmantaaaabbbbBmantaaaabbbB
mantaaaabbB	mantaaaabBmantaaaaappppppppppppppppBmantaaaaapppppppppppBmantaaaaappppppppBmantaaaaapppppBmantaaaaappppBmantaaaaapppBmantaaaaaplahBmantaaaaaffffB.mantaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbBmantaaaaabbbB2mantaaaaaasaaasaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapBmantaaaaaappppppppppppppppppppBmantaaaaaappppppppBmantaaaaaappppppB#mantaaaaaapppppbbbbbbbbbbnbbbbbbbbbBmantaaaaaapppppBmantaaaaaappppBmantaaaaaappBmantaaaaaadffddBmantaaaaaabbbbnBmantaaaaaabbbbbbBmantaaaaaaappppppppppppppppppBmantaaaaaaapppppppppppppppB%mantaaaaaaaapppppppppppppppppppppppppBmantaaaaaaaappppppppppppppBmantaaaaaaaappppppBmantaaaaaaaapppppBmantaaaaaaaafBmantaaaaaaaabbbbbB!mantaaaaaaaaappppppppppppppppppppBmantaaaaaaaaappppppppppppBmantaaaaaaaaapppppppppppBmantaaaaaaaaabbbbbBmantaaaaaaaaaappppppBmantaaaaaaaaaapB!mantaaaaaaaaaaappppppppppppppppppBmantaaaaaaaaaaappppppppppppppBmantaaaaaaaaaaappppBmantaaaaaaaaaaaapBmantaaaaaaaaaaaabbbbbbbbbbbbbbBmantaaaaaaaaaaaaapBmantaaaaaaaaaaaaabbbbB"mantaaaaaaaaaaaaaappppppppppllppppBmantaaaaaaaaaaaaaappppppppBmantaaaaaaaaaaaaaapB"mantaaaaaaaaaaaaaaapppppppppppppppBmantaaaaaaaaaaaaaaappppppppppppBmantaaaaaaaaaaaaaaappppppppppBmantaaaaaaaaaaaaaaappppBmantaaaaaaaaaaaaaaapBmantaaaaaaaaaaaaaaaapppppppppppBmantaaaaaaaaaaaaaaaaappBFmantaaaaaaaaaaaaaaaaaappppppppppppppppppppppppppppppppppppppppppppppppBmantaaaaaaaaaaaaaaaaaapppppppppB&mantaaaaaaaaaaaaaaaaaaapppppppppppppppBmantaaaaaaaaaaaaaaaaaaapB4mantaaaaaaaaaaaaaaaaaaaaapppppppppppppppppppppppppppB$mantaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbBmantaaaaaaaaaaaaaaaaaaaaaappppB$mantaaaaaaaaaaaaaaaaaaaaaaapppppooooB%mantaaaaaaaaaaaaaaaaaaaaaaaaappppppppB!mantaaaaaaaaaaaaaaaaaaaaaaaaappppBmantaaaaaaaaaaaaaaaaaaaaaaaaapB!mantaaaaaaaaaaaaaaaaaaaaaaaaaaappB mantaaaaaaaaaaaaaaaaaaaaaaaaaaapB#mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapB$mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapB&mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapB/mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapppppppB5mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapppppppppppBGmantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaapppppppBmantaaaaB	manstapppBmanstabpBmanstabbbbbBmanstabbBmansiwnqjwbsjwjqbwnBmanrabbbbbbbbbbbbvbvvvvvbbvvvB1manrabbabajskdidjidjxhdhxhjdhdhdidhdbdgsijsgsushdBmanraaaappppBmanntttaaaaaaapppppppBmanntapBmanntabB	manntaappBmanntaaapppB#mannnttaaaaaaaaaaaaaaaaaaaaapppppppBmannntappppBmannntapBmannnnttaaapB
mannnnntepBmannnnntaaaaaaaabbbbBmannnnnnnntaaaaappppBmankyusmankyusBmankanBmanjuuurBmanjurrrrrrBmanjadaBmanjaBmanissBmaningBmaniknyaBmaniiiisBmaniakB	maniaaaaaBmangtabB
mangkuknyaBmangkinBmangkasB
mangcreeppBmangatBmangaatsBmanfapB
manfaatntaB
manfaatkanB
manfaarnyaBmanequinBmanelahBmandirBmandiinBmandeBmandangBmandanBmandalaBmancungBmancinhBmanchapsBmancapBmancaapB	mancaaaayBmancaBmanatepzBmanatbBmanatappppppBmanatabsB
manatabbbbB
manataaappBmanarikBmanapBmanakahBmanajemennyaBmanahanBmanahBmanadoBmanaaaaaaaapB	mamuaskanBmamtappppppppppppppppppppppppppB	mamtappppBmampusBmampetB
mamberikanBmambantuBmamangB,mamajhddhhdjlkhgdbhshdhgstywytststteteetywwtBmamacihBmaluinBmalisaB	malingnyaBmalhBmaleemBmaldivaBmalayBmalaupunBmalasihB
malasahnyaB
malapetakaBmalangBmalampunBmalamnyaBmalaahBmalBmakwiwBmaksutBmaksimumnyaBmaksimalkanB
maksimalisBmaksihmaksihB)maksihhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhBmakroBmaknyuzBmaknyuussssBmaknyusssssssssssBmaknyussssssssB
maknyussssBmaknyossssssB
maknyossssB	maknyosssBmaknyossBmaknB	maklumlahBmakkasihBmakjossssssssssBmakinnnB	makinjayaBmakiBmakesureBmakenyBmakeenyaBmakcihBmakasutBmakasuhBmakassarBmakasiyB.makasiiiiiiiiiiiiiiihhhhhhhhhhhhhhhhhhhhhhhhhhBmakasiiiiiiiiiiiiihBmakasiiiiiiiiiiiBmakasiiiiiiihBmakasiiiiiihhhBmakasiiiiiihBmakasiiiiihhhhhhhhhhhhhhhhB
makasiiiiiB
makasiihhhB	makasiihhBmakasihhjjjBmakasihhhhhhhhjjB9makasihhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhB%makasihhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhB#makasihhhhhhhhhhhhhhhhhhhhhhhhhhhhhBmakasihhhhhhhhhhhhhhhhhBmakasihhhhhhhhhhBmakasihhhhhhhhBmakasihhhhhBmakasigBmakasichBmakasaihBmakaroniBmakanyaaB	makannnyaB+makaksjjjjdjdjsjnnnnsbbsbbvvsvzvbbsnosjebbxBmakaishBmakainBmakacihBmakaciBmakaasiiiihB	makaasiihB
makaasihhhB#makaaaaasssssiiiiihhhhhhhhhhhhhhhhhBmakaaaaaaassssiiiiihhhhhhhhBmakaaaaaaaasiiiiiihhhhhhBmajukanB
majalengkaBmainnyBmainnB
mainfestedB	mainboardBmainanyaBmailBmaianBmahmahBmahkotaB	mahasiswaBmahallllB	mahalanyaBmahakBmahaalB
mahaaallllBmahaaaaaaalllBmahaBmagnetiknyaBmagnaBmaghribBmagentaBmaffBmafBmaenanB
madurasakuBmadurasaBmaduraBmadridBmadikipeB&madevvvvvvvvbvvbbbbvvvvbvbvvvvvvvvvvvvBmaddaBmadahBmadaBmacoBmachoBmachhhBmacetanBmaceetttBmacanBmabukBmabokBmaasBmaantappppppppppppppBmaantaaaapssB
maantaaaapB
maanntaappBmaanntaaaaappppppBmaannntttaaappBmaamkigkBmaalBmaakasiBmaahhhhBmaafnyaBmaaantaaappB
maaantaaapB%maaannntaaaaaaaaaaaaappppppppppppppppBmaaannnntttttaaaaaaapppppppppppBmaaakaaasihBmaaahBmaaaantaaabBmaaaantaaaaappppB!maaaannnnnnntttttaaaaaaapppppppppBmaaaannnnnnntttttaaaaaaaapppB	maaaaatapBmaaaaantapsBmaaaaantaaaaaaaaaaappppppppppppBmaaaaannnntaaaaaapBmaaaaannnnnttttaaaaaabbbssssBmaaaaaantapBmaaaaaannnnttttapppppppppBmaaaaaannnnnnttttaaaaaaaaapppppBmaaaaaaatapBmaaaaaaataaaapBmaaaaaaantaaaapBmaaaaaaaaantaaaaaappppppppBmaaaaaaaaaantaaaaabB$maaaaaaaaaaannnnnnnnnttttttaaavvvvvvB!maaaaaaaaaaaantaaaaaaaaaaaaaaaaapB>maaaaaaaaaaaacccccccccccccccccveeeeeeeeeeeeeeeetttttttttttttttBmaaaaaaaaaaaaantaaaaaapB'maaaaaaaaaaaaantaaaaaaaaaaaaaaaaaaaaaapBmaaaaaaaaaaaaaantafB5maaaaaaaaaaaaaaaaaaaaaaaaaaantaaaaaaaaaaaaaaaaaaaapksBmaaBlycraBlyatBlwalitasBluxBluwesBluvvvvvBluuuummmmayanB	luuummmmmBluurBlututnyaBlutuBlusuhBluruasBlurrBluputBluoaBlunturrrBlunturrBlunturanBlunjungBlungsetBlunbangBlunayanBlunanBlunaB-lumyanlahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhBlumyanlaahhBlumyanaBlumyaaaaaaaaaaaaaaanBlumpurrBlumngyanBlummmmmaayyaaannnnBlummayanBlumixBlumayyyaaaaannnnnnnnnnnnBlumayyyaaaaanBlumayunnnnnjnnjjnnBlumaysnBlumayqnBlumayenBlumayannnnnnnnnnnnnnnnnnnnnnnnnBlumayannnnnnnnnnnnnnnnnnnBlumayannnnnnnnnnnnnnnBlumayannnnnnnnnnnnnnBlumayannnnnnnnnnnBlumayannnnnnnnnnBlumayannnnnnnnnBlumayannnnnnnnB
lumayannkeBlumayanllahB	lumayanlhBlumayanlahhhhhBlumayanlaaahBlumayangBlumayancumaBlumayananlahB	lumayananB
lumayamlahB	lumayalahBlumayahBlumayaanlahBlumayaaaqaaanqBlumayaaannnnnnBlumayaaannnnB
lumayaaannB
lumayaaaanBlumayaaaaaannnnnBlumayaaaaaannnBlumayaaaaaaaaaannnnnnnnnBlumayaaaaaaaaaanBlumayaaaaaaaaaaaaaanB(lumayaaaaaaaaaaaaaaaaannnnnnnnnnnnnnnnnnB$lumayaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBlumatanBlumanyunnnnnnnmnnnBlumanyaB	lumanayanBlumaayannnnBlumaayanBlumaanB	lumaalyanBluluranBlukisBlukayanBlucuuuuuuuuuuuuBlucunyaBlucukkkkkkkkkBlucukkkBluckyBluckerBlubukBlubayanBlubangnaBluasnyaB	luamayaanBluamaaaluamaaaaaaBluamaaaaaaaaaaaBluamaaaBluamaBluaammmmbatBluaamaaaaaaaaBluaaaaaaaaaerrrBltihanBltcB&lsudduudududhdbcfjfjrjfjfrurjrurjjrjrjB%lsndjisskjsjsiiwnwlpajsnmdhajdkdiwkmqBlskowmBlsinBlrnBlrbihBlpaBlowwbatB	lowresponBlowresBlouspekBloubeatBlotusnyaB
lostronganBlosiBlosBloroBloremBlordBlorBlopelopeBloopBlooooorBlooohhhBloooBlooknyB
lookinglahBloohhBlooBlonteBlonjongBlongorBlonglastBlongjonBlongjohnB	longgarinBlondotBloncerBloncengBloncatBlolosBlololB;lolllllllllllllllllllllllllllllllllllllllllllllllllllllllllB	lokasinyaBlokalnyaBlokalanBloiserBloisBlohhhhhhBlohhhhBlohhhBlohhBlogistikBlogisBloginBloeBlodBlockingBlocerBlocalBlobokB	lobangnyaBlobanggBloakanBloadingBloadBlnsaBlnjtknBlngsBlngnsgBlngananBlncarBlnbBlnacarBlmyanlahBlmyBlmnynBlmnBlmayamwBlmaaaBllumayanBllluuuuuuuuummmmmaaaaaayyyaannB
lllllluuuuBllllllllllluuuuummamamamamaBlllllahhhhhBlllahBlliatBllerBllebaranBllambatBllamaBlkpBlkhBlkB	livfancomB	liverpoolBlivelyB
littleponyBliternyaB	literaturB	literallyBliterBlisttttB	listiknyaB	liquidnyaBliptonnyB	lipstiknyBlipsticknyaBlippB
lipetannyaBlipetBlipcreamBlipbalmB
lipatannyaB
liontinnyaB	liontinnyBliontinenyaBliontinBlintingBliningB
lingkunganB
lingkarnyaBlinennyaB	lindunginBlimsBlimeBlimbBlimayanB
lilitannyaBlilinBlikelikeBlikeeBlikedBlihtBlihattBlihatnyaBlihatkanBlightsBlightnyaBligatB
lifeyesnyaBliesBlidocainBlidoBlidahnyaBlidahBlicitBlicinbahannyaBlicensedBlicenseBliatxBliatnyBlianB lhxkyskydkydlhcluhxlyhflydlhdoyxBlhtBlhooooooooooooooooooooooooooB
lhhhhhhhhjBlhhhhhhhhhhhhhhhhhhhhBlhhgkhdkhdkgigxBlhhB
lhaaaaaaaaBlgsubgBlgsingBlglgBlgkapBlgiiiiBlgihhBlggananBlezattBlewtBlewindBlewaBlevihBlevaisBlevBletsBletoiBlesungBlessBlesponBlespaulBlesBleptopBleptoBlepinyaBlepekkkkBlepekBlepaskanBlepasinBlenturlenturBlensnyaBlenshoodnyaBlensanyB
lengkunganBlengkungB	lengketttB
lengketnyaB	lengkeettBlengkaplengkapB
lengkaplahB	lengkapinBlengkaB
lengirimanBlenganyaBlendingBlencirBlemxBlemurBlemperBlempengBlemotttttttttttBlemotttBlemonnyaBlemoniaBlemonBlemnyankelihatanBlemmmmbutttttttttBlemmBlembuutB	lembuttttBlembutlembutBlembuhBlembuBlembagaB
lemayanlahBlemasBlemahnyaBlemahhhB
lelunturanB	lelevisanBlelehanBlelahnyaBlekukanBlekangBlekBlejingBleihBlegendsBlegamBleegingBleeadosBledepanBlecwtBlecillB
lecetlecetBlecehkanBlebuhBlebohBlebinhyaBlebikB	lebiiiiihBlebihhBlebihanBlebeBlebarrBlebariBlebaBlearnBleaBNldlofgplskanzkflgldlskanjajsjlallappalfkfkvkkzlalapqlsmdmxmxmmxllalallaalldmddBldkfndhxdmdmdjddmBlcucktxkhzjkzuckxuvkBlctBlchzkizkBlceknhheBlbjBlazulliBlazizBlazimnyaBlayatBlayannanBlayananxBlayananannyaBlayanBlayBlawangBlavenderBlautBlaurierBlaundryBlaunBlauBlatitudeB
latencynyaBlateBlatalBlarutanBlarutB	larisssssBlarissBlarisnyaB
larislarisBlarinyaB
lariisssssBlarangBlapukBlaptopkuBlaporinBlaporiBlapisinBlapisiB
lapisannyaBlapienyaBlapangBlapakmuB
lapaklapakBlapakkkBlapaaakBlapaBlanyardBlanyaBlantjarBlantaranBlantangBlannyaB	lanjutnyaB	lanjutkenBlanjutkannnB	lanjutganB	lanjotkanBlanhsungBlangungB
langsungdiBlangsubgBlangsngB
langkahnyaBlanggnanBlanggannBlangganannyaBlangganBlangaungB	langanganBlandyB	landscapeBlandepeBlancuarrrrrrrrBlancarrrrrlancarrrrrrrrBlancarrBlancarkandanB	lancarkanB
lancarjayaBlanacarBlamunBlampunhBlampiranBlamoBlammmaaaaaaaaaaaB	lamjutkanBlaminatingnyaBlamiiiiBlambnBlambatttttttBlambatttBlambatsampaiBlambatlambatB	lambatkanBlambaatBlambaaaaatttBlambaaaaaatBlambBlamasihBlamapengirimanyaBlamannyaBlamalamalamalamalamalamamamaB$lamalamalamalamalamalamalamalamalamaBlamalamaBlamakBlamainB	lamabonusBlamabatBlamabanB	lamaaaaasBGlamaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBlamaaaaaaaaaaaaaaaaaaaBlamaaaaaaaaaaaaaaBlamaaaaaaaaaaaaaBlamaaaaaaaaBlaluiBlalubdiBlalalalaB	lalalaalaBlalakBlalagiBlakhhBlakeBlakdmakmiaccBlakaB	lajingnyaBlainzBlainyyaB	lainnyaaaBlainnnyaBlainnBlaineBlainanBlaimBlaiinBlaiBlahmkshB
lahmakasihBlahkBlahirnyaBlahirBlahhlahhhhhBlahhhjjjjjjBlahhhhlahhhhBlahhhhhjhhhjjjjjjjjjjjhhhhhhhhjB"lahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhB%lahhhhhhhhhhhhhhhhhhhhhhhhhhhhhbbbbhhBlahhhhhhhhhhhhhhhhhhhhhhhhBlahhhhhhhhhhhhhhhhhhBlahhhhhhhhhhhhhhhBlahhhhhhhhhhhhhhBlahhhhhhhhhhhhhBlahhhhhhhhhhbBlahhhhhhBlagyBlagsungBlagkBlagipulaBlaginyaBlagingBlagilagiBlagijikaBlagiiiiiiiiiiiiiiiiiBlagiiiiiiiiiiiiiBlagiiiiiiiiiB
lagiiiiiiiBlagidechBlaghBlagendBlagekBlageeeBlageeBlaennyaBlaeeBlaceBlaborisBlaboreBlabaranBlabaBlaamaaaaaaaaBlaahhhhhBlaahhhBlaaahhhhhhhBlaaahhhhBlaaagiBlaaaamaaBlaaaahB
laaaaahhhhBlaaaaahBlaaaaaaaahhhhhBlaaaaaaaaahhhB?laaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaahBlaaaaaBlaaaaBkzoBkzlBkzhBkyrirBkyokoBkyoBkylBkyalBkyaeBkwnyaBkwnkuB
kwlitasnyaBkwkwkwkwkwkwwkkwwkwkBkwkwkwBkwkwBkwkkwBkwitansiBkwciumBkwatirB
kwalitsnyaB	kwalitassBkwalitasnyamaknyussganBkwalitasnyaaaaaB	kwalitaseBkwalitahBkwalitaaBkwalitaBkwaliasB
kwakakakakBkuuuuuuuuuukB	kuuuuuuuuBkuuuuBkuuuBkuupdateBkutirBkutilnyaBkutilBkutekBkutaBkusuttBkusutnyaBkusukaaBkusukaBkuslitasBkusirnyaBkusirBkurusssBkurusnyaB	kuruirnyaBkursiBkurqngBkurisanB	kurirnyaaBkuriBkurasBkuranvBkuranngBkuranhBkurangkurangBkuranggBkuranBkuramgBkurakitBkuraingBkurahBkuragBkuraanggBkurBkuqlitasB
kupingnyaaBkupilihbselainBkupikirBkupasBkunsumenB
kuningbsolBkuninganB	kuniiiingBkungtayB
kunciannyaB	kumuwlohuBkumpulanBkumilikiBkumilBkumelBkumatBkumanBkumahaBkultsBkulittBkulitnyBkulitasxBkuliatBkuliahanBkulakBkulaBkukitBkukiraBkukarBkujanggrosirBkuisB
kuinginkanBkuharapBkugakBkugaBkuerenBkuenyaBkuennceenggBkuencengBkuecilBkudhuBkudapatBkudangBkucobaBkuciwaBkucingkuBkucewaBkucellBkucekBkucariBkucanggBkuburanBkubukaBkubloB	kuatlitasBkuatkuatttttBkuatirBkuantitinyaBkuanBkualtsBkualtiasnyaBkualtiasBkualitsBkualitassssssssssssBkualitassssB	kualitassBkualitasnyaaaBkualitasnnyaBkualitaslahB
kualitanyaBkualitahBkualitadB	kualitaasBkualitaaBkualisBkualiatsnayBkualiatasnyaB
kualiasnyaBkualatBkuakuiBkuaitasBkuaattttttttBkuaaatBkuaBktvBktrmBktpBktnyBktnaBktnB
ktinggalanBkterimaBktaxBktanyBktangnBktaB	ksluruhanBkslhnBkslahanBkslBksksbsksvsjsBksinihBksikBksihnyaBksiBksesuaiB	ksempitanB
krwatifkanB	krudunganBkrsBkrrennBkrotoBkrossBkroscekB	kronologiBkromBkrnnyaBkrnapaBkrmnyaBkrmnBkrminB
krmbalikanBkrlBkrjasamanyaBkrjanyaB	kriyukpukB	kriukriukBkritikanBkristalBkrisbowB	kripiknyaBkrincingannyaBkrimnyBkrimanB	krieyeriaBkrgpasBkreteriaBkresesBkrepsnyaBkreoBkrenzBkrensBkrennnnnnnnnBkrennnnBkrennnBkremesBkrekkkBkrekBkreennnBkreeeenBkreeeeeeeeeeeeeeeeeeeeeeerenBOkreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeenBkredibilitasnyaBkreasiBkreamBkrdusB	krdungnyaBkrbekBkranBkramBkrainBkrahBkqrenaBkprcyaanBkpnginBkplitBkpinginB
kpanjanganBkpalanyaB	kotoranyaB
kotorannyaBkotopBkotekBkotakuBkotaknyabsampaiBkostumerBkostBkosmetiknyaBkosanBkoruptorBkorupsiB
korsletingB	korsetnyaB
koreksinyaBkorekinBkorbanyaBkoponganBkopongBkoplokBkoplingBkopikoBkopianBkopiahB	kopetitifBkootorBkoorperatifBkoordinatnyaBkoordinasinyaB
koordinasiB	koordinasBkooperativeBkooperatifeB
kooperarifBkoooooB
koodinatifBkooBkonveksinyaB
kontreksinB	kontrasssB	kontrasnyBkontraBkontorB	konsumsinBkonsumerB	konsumentB
konsultasiB	konsekwenB	konsekuenBkonplainnyaBkonnectBkonjadiBkonidsiB
konfurmasiB	konfrmasiB
konformasiB	konfirmsiBkonfB	koneksinyBkonectB	konecktorB	konditionBkondisixB	kondisineB
kondisikanBkondiisiBkonciBkonaweBkonaumenBkomsumenBkomsenB
komputernaBkomponennyaBkomplrnBkomplitkomplitBkomplineB	komplenanB	komplaineB	komplaiinBkomplaiBkompkainB
kompirmasiBkompetitifkompetitifB
kompermasiBkompenB
kompatibleBkompasBkompalinBkomlaiBkomitBkomisiBkomfirmasikanB
komfirmasiB	komentnyaBkomentatornyaBkomentatB
komentarinBkomentarbkayajBkomennBkomeninBkomeeeeeeeennnBkombinasikanBkomandanBkomaBkolornyaBkolongBkologiniBkolarBkokpaBkokonyaBkokoBkokkokBkokkBkokaBkoitBkoilnyaBkoilBkogxhgfvbjbcxfggkxsjhgBkoenksiterputusBkoenBkoelksiB	kodratnyaBkodisiBkodianBkodenyaBkodeeBkodachiBkocokBkochBkocarBkobtebalBkoboyBkoasB
knyataanyaBkntongBknsumnBknowlahBknockBknoB
kneepadnyaBkneBkndsiBkndorBkndlaBkndisiBknclongBknanBkmsnBkmresekB	kmngkinanBkmjaBkmiBkmhalsnBkmeranyaB	kmeracctvBkmeraBkmenurutBkmbangBkmayanB	kmasannyaBkmajuanBklxBklwrBklubB	klualitasBklpunBklpBklotokB	klonenganBklolBklmaanBklllllkBklipnyaBklipangBklihatannyaBklietanBkliatnBkliatamBklianBklemnyaBkleinBklebihanBklebaranBklaupunBklatenBklasBklaimnyaB
kkuranganxBkkuatanBkkuB
kklualitasBkkkkklkkkkkkkkkkBkkkkkkkkkkkkkkkkkkkklkkkkkB)kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkBkkkkkkkkkkkkkkkkkkkkkkkkkkkBkkkkkkkkkkkkkkBkkkkkkkkkkkkBkkkkkkkkkkkBkkkkereeeeeeeeeeeeeereenBkkkaaaassseeeeeppppppBkkkBkkjghjklhhjkkhvcsfjkmbggBkkerenBkkeeeccceeeBkkecwaBkkecilanB:kkdgighkdkfjfjcjcjfcjfndnfjdhfjfhxisisidpsisogkfjcnvjfhfjfBkkclanBkkcilankkcilanBkkantorBkkanBkkajabahavbaajamkaoanaBkkaB	kjwusppelBkituBkitnyaB	kitimanyaBkitanyaBkitaaB
kissbeautyBkissBkirrBkirqinBkiromBkirmnyaBkirklandBkiringBkirinBkirimnyB	kirimnaaaBkirimmyaBkirimmmB
kirimkirimBkirimkannyaBkirimeBkirimanyB	kirainadaBkiraaBkintilBkinianBkingmaBkindlyBkindisiBkindaBkinBkimpullBkimochiB
kilogebrusBkilaunyaBkilattBkilaaatBkilaBkikirimBkiilatBkiiiBkidzBkidsBkiddingBkiainBkhususyBkhoiroB	khimarnyaBkhawtirB	khawatirrBkhatirknB	khasiatyaB	khasiatnyBkhanBkhakyBkhairBkhabisanBkggaBkgdeanBkfzdhgB	kfogohohoBkfastBkeygenBkeyerimaB$keyennnnlahhhhhhhhhhhhhhhhhhhhhhhhhhBkeyboardnyaBkeyboarBkeyannnnB
kewanitaanBkevewaBkeunguBkeulangBkeukuranBkeukeurBkeujananBketusBketukarBketukangBketujuhBketsBketrnganB	ketranganB	ketombeanB	ketokonyaBketokoBketmptBketisuB	ketipisanBketipB	ketindissB	ketimbamgBketilepBketidaksengajaanBketidaknyamananBketidakcocokanBketianB
ketetanggaBketerngnBketerlbatanBketerlambatannyaBketerlaluanBketerlalaunB
keterawangB
keteranhanB	keterangnBketeranganyaBketemanB	ketelingaBketekukBketekenBketawanBketawainBketaraBketapangB	ketangkapB
ketamvananB	ketakutanBketajamannyaB	ketahiganB
ketahananyBketagianBkesuluruhanB
kesukaankuB
kesluruhanBkesisiniB	kesininyaB	kesingnyaB
kesinginanBkesingBkesinBkesewaB	kesenggolBkesengajaanB
kesenanganB
kesemuanyaBkesembuhannyaBkeselyruhanBkeseluruhanyaBkeseluruhaanBkeselurahanBkeselnyaBkeselamatanB	kesekolahBkeseimbanganBkesehatannyaBkesehatB	kesegaranBkesedotBkesedianBkesederhanaanBkesebarBkesbyBkesasarBkesanyaB	kesananyaBkesamaanBkesalahnBkesalahannyaB	kesajahanBkesahB
kesabarnyaBkerwtB
kerutannyaB
kerupuknyaB
kerumahnyaBkerumahhhhhhhBkerudunhnyaB
kerudungnyBkersamaBkerrrrreeeeennnB	kerrrennnBkerrnnnnnnnnnnnnnnnnnnnnnBkerrnBkerreenB
kerreeennnBkerreeenB
kerreeeeenBkerreeeeeennnBkerreeeeeeeeeeeennnnnnnnnBkeroppiBkeropiBkerokinBkerokanBkermhBkerjoBkerjasamsnyaBkerjakerjakerjaBkerjakanBkerjadamamyaB	kerjaanyaB
kerjaannyaB	kerjaankuBkerjBkeritingBkeriputB	keringnyaB
keringetanBkeringatkeluarBkerikilBkerijekB	keridhoanB	keretakanBkeresekBkerentopmantapselaluorderBkerensBkerennyaBBkerennnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnB+kerennnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnB$kerennnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnB"kerennnnnnnnnnnnnnnnnnnnnnnnnnnnnnB kerennnnnnnnnnnnnnnnnnnnnnnnnnnnB%kerennnnnnnnnnnnnnnnnnnnnnnnnnnbnbnnnBkerennnnnnnnnnnnnnnnnnnnnnnnnB'kerennnnnnnnnnnnnnnnnnnnnnnmnnnnnnnnnnnBkerennnnnnnnnnnnnnnnnnnnB/kerennnnnnnnnnnnnnnnnnnmnnmmmnnmmnnnnnnnnnnnnnnBkerennnnnnnnnnnnnnnnnnBkerennnnnnnnnnnnnnnnBkerennnnnnnnnnnnnnBkerennnnnnnnnnnnnBkerennnnnnnnnnnnB1kerennnnnnnnnnmnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnB#kerennnnnnnnnnkerennnnnnnkerwnnnnnnBkerennnnnnnnnbgttttB"kerennnnnnnakakqlqlqajajajajajsjjwB
kerenlahhhBkerenlaaB#kerenkerennnnnnnnnnnnnnnnnnnnnnnnnnBkerenkerennnnnnnB-kerenkerenkerenkerenkerenkerenkerenkerenkerenB(kerenkerenkerenkerenkerenkerenkerenkerenB#kerenkerenkerenkerenkerenkerenkerenBkerenbnBkerenbingitsBkerenaBkereennnnnnnnnnnnnnnnnnnnnnnBkereennnnnnnnnnnnBkereennnnnnnnnnnmmmmmmBkereennnnnnnnnBkereennnnnnnnBkereennnnnnnBkereennnnnnB
kereennnnnB	kereennnnB	kereenlahBkereemnnBkereeernBkereeennnnnnnnnnnnnnnnnnnnBkereeennnnnnnnnnnnnnnBkereeennnnnnnnnnnnnnBkereeennnnnnnnBkereeennnnnnBkereeenkereeenB
kereeeerrnBkereeeennnnnnnnnBkereeeennnnnnB	kereeeennBkereeeeennnnnkereeeeennnnnBkereeeeennnnBkereeeeeeweeeeeeennBkereeeeeennnnnnnnnnnnBkereeeeeennnnnnBkereeeeeeennnnnnnnnnBkereeeeeeeennnnnBkereeeeeeeeennnnnnnnnnnBkereeeeeeeeenB#kereeeeeeeeeeennnnnnnnnnnnnnnnnnnnnBkereeeeeeeeeeennnnBkereeeeeeeeeeeennnnnnnnBkereeeeeeeeeeeennnnB"kereeeeeeeeeeeeeennnnnnnnnnnnnnnnnB!kereeeeeeeeeeeeeennnnnnnnnnnnnnnnBkereeeeeeeeeeeeeennnnnnnnnnBkereeeeeeeeeeeeeennBkereeeeeeeeeeeeeeeeenBkereeeeeeeeeeeeeeeeeennnnnnnBkereeeeeeeeeeeeeeeeeeenBkereeeeeeeeeeeeeeeeeeeeeenB#kereeeeeeeeeeeeeeeeeeeeeeeeeennnnnnB"kereeeeeeeeeeeeeeeeeeeeeeeeeennnnnB kereeeeeeeeeeeeeeeeeeeeeeeeeeeenB!kereeeeeeeeeeeeeeeeeeeeeeeeeeeeenB"kereeeeeeeeeeeeeeeeeeeeeeeeeeeeeenBPkereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeenBkereeeeeeeeeeeeeeeeeB	kereannnnBkerdnBkerayonBkerasssBkerassB	kerapatanBkeranjinganB	keranjangBkeranaBkerahasiaanBkeraguanBkerabatBkeraaaasBkeraaBkeraBkerB	keputusanB	keputihanBkepuasBkepraktisanBkeplesetBkepitingBkepinginB	kepinggirBkepikirBkepestaBkeperluannyaBkepercayaannyaBkepercayaabBkepengenB	kepemesanBkepelnyaBkepelBkepekaannyaB	kepeeeeetBkepasB
keparinganBkepapalBkepantaiB!kepanjanganbdhsnsjsnsbsbshshshsgsBkepalaaB	keonginanBkenyatanBkenyangBkenyamananyaBkenyamanBkenwoodBkenvicBkentrungBkentelBkenkoBkenionBkenextB
kenerjanyaB	kenderungBkendariB	kencenginBkencanaBkenasanB
kenanggapiBkenanganBkenaliBkenalanBkenakanB
kemrincingB	kempinskiBkempesanBkemnaBkemirpanB	kemirinyaBkemesanBkemekBkemejanyBkemedsosBkembaranB
kembangnyaBkembangkankembangkanB
kembangkanB	kembalianB	kemasukanBkemasnBkemasaanBkemanfaatanB
kemanamanaBkemakanBkemahBkemabaliBkeluhkanBkeluhaanBkeluhBkeluatanB	keluarkanBkeluaranB	kelolosanBkelitasB	kelistrikBkelipatBkelipB	kelimanyaBkelilingnyaBkelihayannyaBkelihatanyaBkelihatannyB
keliatnnyaB
keliatanyaB	keliataneBkeliahatannyaBkelepasBkelengkapanbyaB
kelembutanB	kelelahanBkeleknyaBkelekBkelebihannyaB	kelebaranB	kelawasanBkelapB
kelanjutanB
kelancaranBkelambuBkelalaiannyaBkelakuannyaBkelakuanBkelainB	kelaiatanBkelBkekurangBkekunciBkekuhanB
kekonsumenB	kekocakanBkeknyaBkekirimB
kekiniannnB	kekihatanBkekerasannyaBkekerB
kekentalanBkekencenganBkekecilnB	kekecilinB	kekecilenBkekecilannnBkekecilandehhhB	kekecilamBkekecewaaanBkekcilanBkekardusBkekalimantanBkejunyaBkejualB
kejernihanB	kejenkangBkejelasannyaBkejebakBkejarBkejambiBkejaitBkeisiB
keinternetB
keinginnanB
keinginanqB)keinginannnnnnnnnnnnnnnnnnnnnnnnnnmnnnnnnBkeinginannnnBkeinginankuB
keinginanaB	keinginabBkeinginaBkeinginBkeingetBkeiginanB	keiginaanBkeieBkehitungBkehidupankuB
kehangatanBkehalangBkegusurBkegulungBkegoresBkegitarnyaaB	kegenyangBkegenjetB
kegendutanB	kegelapanB	kegedeeanB	kegedeaanB
kegedeaaanBkeganjelBkegangguBkeereenBkeereeeennnBkeereeeeeeeeeeeeenB
keennaaaaaBkeemasanBkeeerrreeeennnnnnBkeeerrreeeennnnBkeeerreeennnB
keeereeennBkeeeeyyyeeeeeeeeeeeeeeeennnnnnBkeeeereeeeeenBkeeeerB keeeeerrrreeeeeennnnnnnnnnnnnnnnBkeeeeereeeennnBkeeeeereeeeeeeeenBkeeeeeerrrrrrnnBkeeeeeeerrrreeeeeeeeeeeennnBSkeeeeeeeeeeereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeennnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnBkeeeeeeeeeeecilBkeeeeeBkeeeeBkeduakalinyaB	kedtanganBkedpnnyaBkedpnBkediriBkedipnyaB	kedeteksiBkederBkedepanxBkedepanaBkedelaiBkedatangannyaBkedapaB
kedalamnyaBkedalaBkecwaaBkecutBkeculBkecukurBkecukiB
kecolonganBkecohB	kecocokanBkecoaaBkecmsnBkecjlB	kecilnyaaBkecilnyBkecilllllllllBkecillanBkecilkanBkecildiBkeciilB	keciiilllBkeciiiillllBkeciiiiiiiilllllBkeciiiiiiiiiiiiiiiiiiiiiiiiiiilB
kecewwaaaaBkeceweanBkeceweaBkecewavparahBkecewakecewaB	kecewakanBkecewainB/kecewaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaB*kecewaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaB%kecewaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaB#kecewaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBkecewaaaaaaaaaaaaaaaaaaaBkecewaaaaaaaaaaaaaaaaaBkecewaaaaaaaaaaaaaaBkecewaaaaaaaaaaaaBkecewaaaaaaaaaaBkecewaaaaaaaaBkecewaaaaaaB
kecermatanB	kecepetanBkecelikBkeceleBkecelahBkecehhB	keceewaaaBkeceaBkeccehBkecapnyB
kecantikanBkecanggihannyaB	kecamatanBkecamataBkecabutBkebwahBkebumenBkebulBkebuktiBkebukalapakBkebuangBkebonB
kebohonganB	kebnyakanB	kebnjiranBkebiruB
kebingungnB	kebijakanB	kebiasaanBkebiasaBkebgianB	kebetukanBkebesranB	kebesarenB	kebesarabBkebesaraaaaanBkeberuntunganBkeberhasilanB	keberatanBkeberadaannyaB	kebeneranBkebenarannyaBkebeletBkebelerBkebelB	kebeaaranBkebaretBkebanyakB	kebantingBkebankBkebanggankuB
kebangatanB	kebandingB
kebahagianB	kebagusanBkeaousBkeangaktBkeamanannyaBkealamatBkeakuratannyaB
keakuratanB	keajaibanB	keadaanyaB
keadaannyaBkeadaaanB	keabsahanBkeaadanBkdpnnyBkdpanyBkdjhddbmamamzxxbsbbansnnB kdjhdchchchcbcbfbcbxbxbxbxbxhchcBkdepanxBkdepanB	kdengeranBkdcBkcsBkcptnBkclinBkcilanBkcewainBkcewaaaBkcBkbukaBkbtulanBkbrkahanBkbrinBkbriB
kbrhasilanBkbpsBkbnyknBkbnyakanBkbmBkbijaksanaanyaBkbesranBkbacaBkayuuuuuuuuuuBkayuneB	kayangnyaBkayaklBkayakeeBkayagkBkawannyaBkaulitasB	kaualitasBkatunnyaBkatuBkatsironBkatroBkatritBkatnaBkatilayuBkatikaBkataraBkatannyaBkatBkasutBkasuhBkastamerBkassiihBkassiatyB	kasperskyBkasosBkasmirBkasirBkasiiiiiiiihhhB	kasiiiiihBkasiiiihBkasiiihhhhhBkasiihhBkasiihBkasihnyaBkasihlahBkasihhhhhhhhhhhhhhhhhhhhhhBkasihhhhhhhhhhhhhhhhhhhhhBkasihhhhhhhhhhhhhhhB
kasihhhhhhB	kasihhhhhBkasiahBkasehBkaseBkascingBkasatB	kasarrrrrBkasarrBkasaihBkasB
karyawanyaBkaryanyaBkaruBkarokeanBkarokeBkarnakanBkarnahBkarnaaaaBkarnBkarlBkarisBkarirmuBkarettBkarensB	karenakanBkardussBkardusnyBkardusinBkardungBkarbonB	karawachiB
karatannyaB	karaokeanBkarangBkaranaBkaranBkarakterkahB	kaputanyaBkaputBkapukBkaptenBkaprahBkapooookBkapokinB	kapasitisBkapasBkapanpunBkapamBkapaBkaosnyBkaoskakiBkaokB
kantungnyaBkantongyB	kantongkuBkannBkanjiBkanggurunyaBkanggeBkangenBkaneboBkandungannyaB
kandangnyaBkancutB	kancinginB	kancinganBkanapaBkamxiaBkamvrettB	kamuflaseB	kampunganBkamprettttttBkamprBkampanyeBkamisnyaBkamernyaBkameraneBkamdigB	kamaranyaBkamaBkaltengBkalselBkalpBkalouBkaloriBkalonadaBkalokBkalkulatornyaBkaliyaBkalixBkalimayaB
kalimantanBkaliiiiiiiiiBkalihB	kalibrasiBkalenjiB	kalengnyaBkalengnyB
kalengggghBkalenggBkalenderBkaleeeBkalbenyaBkalbeBkalawBkalaubteB
kalahbsamaBkalaBkaktusonlineBkakixBkakinyBkakekBkakehenBkakasayaBkakaknyaBkakakakaBkakaittBkajnB8kajagzvxhdbdjdnjxjxnxjfnxnxnxnxbjsndjdnxbjxndjdnxjdnjfnfBkaisarBkairB
kahujannanBkahoyongBkahiyangB
kahitannyaBkahawatirkanBkahareupBkahaBkagetnyaBkagahBkagaaaBkaftanBkafangBkafanBkaesangBkaenyaBkaelBkadungBkadarBkadanagBkadanB
kadalurasaBkadalauarsaBkacoBkacirB	kacingnyaBkacawBkacauuuuBkacauuBkacaukacaukacaukacaukacauBkacapitB	kacangnyaBkacanginBkacamatsB	kacaauuuuB	kabupatenB	kabelnyaaBkabelnyBkabelmyaBkabekBkabariBkabakabaBkaauBkaamiBkaaaaakBkaaaBjzlvlcjxlvkzgBjygaBjyaBjxiBjwjwbwbananananbasbbssbsbbddbdB
jwhsgshahaBjwbannyaBjwbanBjwabnyaBjwabBjvcBjuwaraaaaaaaBjuveBjuuoooossshBjutekBjutaBjuslBjurganBjuranganBjurahaaaaannnnB
juragannnnBjuraganeBjuragaaaaaaaanBjupratBjuozzzBjuosssssssssssssssBjuosBjuoossBjuoooozzzzzzzzzzBjuooooossssBjuntaiBjunguBjungB
jumsuitnyaBjumssuitBjumpswitBjumpsuitnyaBjumpBjumlahxB	jumlahmyaBjulanBjujuuuuuuuuuuuuuuuurBjujurpelayananBjujurlahB
jujurannyaBjuhaBjuguBjudgeBjudeessBjualnBjualijBjualanyBjualannyBjssnsnsjsnsnsjdnsnsjdkdnsnsjsiBjskakabwBjsjskssB>jsjaiaoawoeoeeldmzpdpdskahahuahnwnwosksmddjhusssnkskasmnsnsbssBjsisisisB jshdusmsmjeyshslsngislanzvwujwkwB8jshdjdofosgsiwoaopskdhskskagsudodlahaoakshdifldlshsjdodkBjshdbdB'jshaomnzvsjskskiclqpanzvzkaajdudjdjsjksBjsaB"jrojosssssssssssssssssssssssssssssBjrngBjrenggBjreeengBjreeeengBjrakBjrabutBjpegBjozzzzzzzzzzzzzzzzzzBjozzzzzzzzzzzzzzzzBjozzzzzzzzzzzzzzB,jozzzzzzzzzzzzxsssssssssssssssssssssssssssssB$jozzzzzzzzzzjozzzzzzzzzzjozzzzzzzzzzBjozzzzzzBjozzzzzBjoztB
joystiknyaBjousssssBjossssssszszzzzzssssssssssBjosssssssstttttBkjosssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssB[josssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssB8jossssssssssssssssssssssssssssssssssssssssssssssssssssssB4jossssssssssssssssssssssssssssssssssssssssssssssssssB/josssssssssssssssssssssssssssssssssssssssssssssB'josssssssssssssssssssssssssssssssssssssB&jossssssssssssssssssssssssssssssssssssB jossssssssssssssssssssssssssssssBjossssssssssssssssssssssssssssBjosssssssssssssssssssssssssssBjosssssssssssssssssssssssBjossssssssssssssssssssBjossssssssssssssBjosssssssssssssBjossssssssddddssssB
jossssssssB
jossssslahB	josssssaaB	jossssaasBjosssddBjosslahBjossjossssssBjosshtBjossaBjoshhBjosddBjosdahhhhhhhhhhhhBjorannyaBjoozzzzzzzzBjoosssssssssBjoosssssBjoossdBjooossssslahB	jooosssssBjoooozzzzzzzzB%joooossssssssssssssssssssssssssssssssBjoooosssssssB
joooosssssBjoooosB3jooooozzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzB&jooooosssssssssssssjooooosssssssssssssBjooooossssssBjoooooosssssssssssBjoooooossssssssB	joooooossBjooooooosssssssssBjoooooooossssBjoooooooooooosssssssssssBjoooooooooooosBjooooooooooooooozzzzzzzB0joooooooooooooooooooooobjoooooooooooooooooooooobB
joooooobbbBjonkkkBjonggolBjonbBjombangBjoknyaBjojuB	joissssssBjogyaB	jogvjlpppBjogloB
joggingnyaB	joggernyaBjogedBjoesBjodihnyaBjobssssBjobssBjoblahhhhhhhhhhhhhhhBjobbsBjobbbbsB%jobbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbBjobbbbbbbbbbbbbbbBjobbbbbbbbbbbBjobbbBjoasBjoBjnternetBjnsBjnrBjnhBjmlhBjmlahnyaBjmkmBjmkkknBjmbtkotlpppqBjlekkBjketnyBjkaB
jjuuijjhggBjjujurBjjugaBjjsnsjbssnajajBjjsBjjosssssmnnnnnnnBjjooooossssssssssssBjjjoosaaaaaaasaassssssssB
jjjoooonnnBjjjkchguiihhpphugffhioBjjjjjjBjjjjakakakakakaBjjgffhklhdsafhurdvnkirdchiuedgBjjgdnkigdgjjjB
jjdjdkfkjfBjiwoBjiwahBjiwaaaaaaaaaaaaBjiwaaaaaaaaaaB	jiwaaaaaaB	jivvvvaaaBjirrrBjirpdljidpelejjeisksjsuBjirimBjirBjiosnhjsBjinjingBjinhaoBjinghaoBjikatBjikalauBjijikBjiiooossBjiiiBjidatnyaBjidatB	jiaaaannnBjiaBjhvfssvnBjhtnBjhossB5jhjkjhhhhhshsjjsjsjsjsjsjjsjjsnsjsnjsnsjsjjsjehsjshheBjhitnB
jhhjhjbhghBjhewaBjhaBjguaBjgncumaBjgkBjgjgBjggBjgdfB2jfvccnkmbugddxvbjklpibhnfyyinhjknvcdszxvnnnnmbggjjBjfjfjfjBjffjBjfBjeuduudjdjfBjetisB	jeroannyaBjerningBjerniiiihhhhB
jerniiihhhB	jernihnyaBjerniBjeraB
jepretanyaBjepratBjeparaB	jepangnyaBjeniusBjenihBjeniferBjenggottnyaB
jenggotnyaBjenengeBjenengBjendolanBjenasB	jempolnyaBjempolllBjembutBjemasanBjelyBjelwkBjelonhBjelnyaB	jellycaseBjellyBjellllllllaaaaaaassssssssBjellllekkkkkB	jeleksayaBjelekkkkkkkkBjelekkkkkkkB	jelekkkkkBjelekkkkBjelejBjeleekkB	jeleeekkkBjeleeeeeeeeekkkkkkkkkkkkkkkkkBjeleeeeeeeeeeeeeeeeeeeeeeekBjelassBjelasnyaBjelaslahBjeknyaBjeketnaBjeketBjekekBjekeBjeidoBjeggerBjeeleeeeeeeekBjecilBjebulBjebollllBjebollBjebawBjeanspoBjeanBjdxBjdwlBjdwalBjdnyBjdmaljhxakaihdbdkaahxujwnxnicaaBjdlsB-jdkshisbfhanizbsidnxbzoandinaiansudidbdhsjsksB%jdksgchsishdgidhdhdovdodhdosvdodbsobdBjdjsnsB	jdjidjdjdBjdinyaBjdinyBjdikanBjdiiBjdigknyeselbelinyaBjdidjsisjdjB$jdhdhdklazvzbzjizvsnzoskbznznjzjzbxbBjdeBjdbdbB	jdbbbdjsjB	jcfhbbcxcBjbchbgjB
jazakillahBjazaBjayapuraBjayalahBjawabxBjawabaneB
jawaaabbbbBjauuuuuuhhhhhhBjauuuuuhhhhhB
jauuuuuhhhBjauuhhhhBjauuhBjauhhhhhBjauhanBjaubBjatuhanBjatiwaringinBjatengBjassnyaB	jasketnyaBjasanyaBjarumaBjarjB	jaringnyaB
jaringanjfBjareneBjareBjarakternyaBjarahBjarBjapcheBjapchaeB	jaozzzzzzBjanuariB	jantunganBjannahBjanjkanBjaninBjanhanBjangkrikBjanggalBjangaBjaneBjancokBjanaganB,janabzgxysvavzjxjsjshxgshabsbxhxusjsjsbxgxusBjamxB
jamntanganBjamiinBjambuBjambngBjamaahBjalurnyaBjalurBjalukBjalnBjalaniB-jakkkksbsbsuhshsgsgssansbsvsvzgzhbsbsbsbshshsB	jaketnyaaBjaketnyB	jaketnnyaBjakbarBjakBjajsbsnsmsjsisnsvshB$jajapapqnabzhaiapmansuajnaanwnjsiwjwBjajanananananananananananBjajajjbvBjaitanbkurangBjaitamBjaitaBjahutanBjahittanBjahitannuyaB
jahitanntaBjahitannnyaBjahitanbrapiB
jahitanayaBjahitaanBjahatBjagonyaBjaganBjadulnyaBjadoelBjadinyBjadinpermslahanBjadiiBjadebotabekBjaddiBjabitanBjababekaBizinkanBiyyaaBiyyaBiyiiiiiiuuuuuiuuuuBiyhBiyalahBiyakanBiyaaaaaaaaaaaaaaaaaaaaaaaBiyaaaaaaaaaB	iyaaaaaaaBiyaaaBiyBiwatchBivoryBityyyyyymdkaosuzhznkaoajBityB	ituuuuhhhBituuBitusajaB
itungannyaBitukanBititBitikBithBitasBitamBitaliaBistroBistriquBistripunBistrinyaB	istimiwirBistimewanyaBistimewalahBistimewaahhhBistimewaaaaaBistimewaaaaB
istilahnyaBistickBisterikuBistBisoBislamiBisixBisinnyaBisikanBisijklhgyfdlkhttrfgBisiinBironBirisanBirigBiqfitnyaBipunBiptvBipsumBipisBipinBinyakanBinyabaguessssBinveBinulBinukBintruksiB
intructionBintixBintexBintetnetBinternetnyaB
interaktifBinterB
intensitasBintensifBintanBintalB
insyallohhB	insyaallhB	insyaalahBinsyaaallahBinsyaaB	installerB
installasiB	installanB	instagramBinssyaB	inspirasuB	insolenyaBinsoleBinsolBinsidentBinseBinsayaB	inovationBinovaBinoBinniBinnernyBinlineBinkraisernyaBinkBinjekBinjakB	inisyatifBinisayaBininyaBininBinimalahBinikanBiniiniBiniiiiBiniiB
inidonesiaBinibarangnyaBingunkanBingrisBingkangB
inginkannnBinginkaninginkanBingiinBingetnyaBingatanBingBinfrarednyaaBinframerahnyaBinfornativeBinformationBinformasikanBinfonyBinfocusBinezBinewsBinerBindustryBindustriBindroBindraB
indovisionBindotopiBindosialB
indonsesiaB	indonesisBindividuBindisBindikatotnyaBindihomeBindihbbarangBindehoyBindahnyaBindBincloudeBincldB
incididuntBinbokBinatructionBimuuuuutBimuutBimutttBimutsBimutnyaBimsBimpiankuB	immaannyaBimlekBimingnyaBimbunBimangBimaiBilopuBillfeelBilkanBilikeBilfilBilehBilegantB
ilaannggggBikurBikuBikqtBiklhasBiklasinBiklanyaBiklannyaenjanjikanB
iklanbukanBiklamB	iklaannyaBiklaBikieBikhlasiaBiketannyBikeBikatnyaBikatannyBikanyaBikalnBikalanBikaBikBijooBiituBiiiyaaaaaaaaBiiiiiiiipppB&iiiiiiiiiiuiiiiiioiiiiiiiiiiiiuiiiiiiiB7iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiipB
iiiiiiiiiiBiiiiBiiihhhBiiiBiiaBihihihiBihhhBigieviegogeoheigeBiflixBifikatBifikasiBifBieuBiesBiemBiedBidrB1idjsjsjjdpsosidjjjfjfjjfjfjjfufjhsggyshwdhdjdjdjdB	idikattorBidemBidamBicuciBiconicBiconBicomBichBicelandBiccBibunyaBibundaBibukotaBibadatikBiannyaBianBiaaB9hzjsnsnzjsjdjdjdnixjdbagasjizusbsjzudbixuxnduxhsushsuusbsBhzB
hytdxbjiyfBhyaB%hxxgshjzoswhdgdhnevxvsjwjgxhsuwgxgsjsBFhwnsnsnsmsmsmsmambdbabbagghnsnsnsmamammaammaammamamamamaamamsmsmmssmmsBhwjvuejekknehBhvhjjjccohxhxxxogxucoxgxohvpjBhuwaaB"huuuuuuhhhhhhhhhhhhhhhhhhhhhhhhhhhBhuuuhhhBhuuuftBhuuuffffB
huujcchhjvBhuuhhBhuuhBhutanB	husnudzonBhusniBhurupBhurufnaBhuntingBhumftBhulkBhukumBhujniBhujauBhujannyaBhujaanB
huitscjutrBhuiBhuhuyyyBhuhuyBhuhuuuBhuhuhhhhBhuhuhBhuhhBhuhfBhuftttB	hufthjadiBhuffBhufBhuehuehueheB
hueheueheuBhueheheBhubungannyaBhuayuBhuaweiB	hturnuhunBhtmnyaB	hsusksksiBhsuajjajakakkwkakakakakkakqkqkqBhsrganyaBhsoxjwmalxhwiBhsngssnsnshshabvvsBhslusB?hsjhdhdhdjdhhdhdhdhdhhshshshshhshshshhdhdhdhehhdhdhdhdhhdhdheheBhsisvsiwvsusvBhsilBhsiksbwvakapajbavakknaB hshsjjsjsjsjajajjajsjsksjsjjskskBhshhshshjskurangBhshhBhshdjB
hsclothingB(hsbsbsjsjsbsbshsushhsbsbsjsjsbsbshsushgsBhsbhsBhsbhbBhsbaganBhrzxBhrznyaBhrussBhrusnyaBhrusnyBhrpknBhrngyaBhrgnyBhrganyanjugaBhrgalahBhrgaeBhrgaaaaaBhrapkanBhpkuBhowliteB	householdBhouseBhoursBhotwheelBhottBhotspotBhoreeeeBhopeBhooxBhooooorreeeeeeeeeeeeeeeeeeeBhongfengBhonestBhominisBholyBholeBholdBhokomoroBhokkBhokiBhokBhohohohohohohohehehehehohohoBhodieexBhodBhobyyBhobyBhobbyBhoawalahBhoakBhoaammmBhoBhnyaaB hnhghjlmnbvvvvjkllbbbgcccjjjbbbbBhndbodyBhnayaB8hmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmB$hmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmB#hmmmmmmmmmmmmmmmmmmmmmmhhhmmmmmmmmmBhmmmmmmmmmmmmmmmmmmmmmmBhmmmmmmmmmmmmmmmBhmmmmmmmmmmB	hmmmmmmmmBhlooB	hlasilnyaBhknbxsgjmmxdazjjtthhdzzBhkBhjungBhjhhvBhjBhitzBhitsmBhitemBhitamnyBhitaamBhisterisBhisBhipsterBhippiiwBhingaBhindariBhinBhimalayaBhilngBhillsnyaBhilanhBhilangxBhilanginBhikssssaBhikinhBhijoBhijbB	hijauuuuuBhijauanBhijapnyaBhijabnyBhijaauuuBhiiBhihiiiiBhihihiiiiiiiiiiiiiBhihihihiBhifiBhidupmuB	hidupakanB	hidungpunBhidrolikBhideungBhibuaranBhiasaBhhmmmBhhmmB/hhjkkllllllllllmnhgczzzzfnjjjhgcccfjkkkogfcccbhBhhjkjbccBhhjjkkkkgcfdxdxkknBhhjaBhhiB/hhhuhhhyyhhuhhvxthchbchhgfgjbccdghvxaaedgghhggyBhhhnvcgBhhhmmmBhhhjjjjjBhhhiB!hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhBhhhhhhhhhhhhhhhhhhhhhhhhhhBhhhhhhhhhhhhhhhhBhhhhhhhhBhhhhhhBhhhhBhhheBhhhaaaBhhgggBhhgBhhfhjkihdssxfdffhB hhfffghhhgghhjjgfdffdddgghjdddddBhhfdfBhheheheBAhhaaaaaaaaaaaahhhhhhhhhhhhhhbhbhbbbbhhhhhhhhhbbbbbbbbbbbbbbbbbbbbBhhaBhgggBhgcBhgbBhffghujkijhgyfghBhfB
heuuuuuuuuB	heuheuheuB	hetsetnyaBhetsetBheseBheroesBheroBherniaBhereBherannyaBherBheppyBhentikanBhentiBhensetBhengkyBhendbodyB	hendaknyaBhendakBhemmmmmmmmmmmmBhemmmmBhemmmBhemmhBhembodiBhelpfulBhelpdeskBhelokityBhelokitiBhelmnyaBhellsBhellBhelikopternyaBhelemBhekterB$hejbeiejjhdjbebjejbjriktlllllltnihwgBhehsvsvsB&hehsshvshsjsjsjsjjsjsjsdhhdhshsjsjsjsjBhehhrleB	hehheheehBhehheheB	hehekasihB
hehehhehehB	hehehheheBhehehheeBhehehhBhehehehwBhehehehehehehheB=hehehehehehehehehehehhehehehhehehhehehhehehehehehehehheheheheB(heheheheheheheheheheheheheheheheheheheheBhehehehehehehehehehehehehehehehBheheheheheheeheheheheB	hehehehehBheheheeeeeeeeeeeBheheheeeBheheehehBheheeheeBheheeeeBhehebebeBhegehBheerrBheemmmmBheelsxBheellnyaBheeheBheeemmBheeeeemmmmmmmmmBheeeeeeeeeeeBheeeeBhedseatBhebohBhebattttttttttBhebatnyaBheavyBheatsinkBheaterBheatBheandsetBheamBhealthBheadshetnyaBheadshetBheadsetxBheadsetnyaaB	headsetnyBheadseatBheadlampBheadbandBheaBhdupxBhdsBhdndksoskbdbbxhdjjsBhdminyaB'hdjsjsjwjwkwksmudjsjwjwjwudiwowlwkwkemwB:hdjivjvzuvivuvuvucycuckbzizbosnsonsobsonsosbnsosnsosbosbisB(hdjamamsehebsvsnkakamabavsvsbsbsgsbsjskaBhdiBhdhdhshdudhwhdhhdhejehrsgsbehebBhdggdgshskahwggsksgsvbsbdbdbdbdBhdddBhcfhfhfhdgdhcjfjvBhcdB	hcbvhdjfdBhbvvccfddddfBhbnnbBhayoBhayhBhayaBhawuBhatuurBhaturrBhatirBhatipunBhatimuBhatganyaBhatapkanB	hatapakanBhassanBhasilyBhasilxBhasilnyhasilnyaB	hasilnyaaBhasilllBhasileB	hasiatnyaBhasiatBhasemBharuuuuuuumBharusyaBharusxBharusssB	harussnyaBharuskahBharuseBharusbsettingBharunyaBharsBharrrrgaaaaaaaaaaaaBharrgaB	harnyanyaB8hariiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiBharibiniBharibdatengBharhanyaBhargsBharggaBhargaygBhargayaBhargapunB	harganyyaBharganyhBharganyapunBharganyaharganyaaaaaaBharganyaaaaaaaBharganyaaaaaBhargangaBharganeB	harganayaBhargamakasihB	hargalahhB	hargalaahBhargalaaaaaBhargaharganyaBhargahargahargaaaaaaaaBhargahargaaaahargahargaaaaBhargabterjangkauBhargaaaaaaaaaaBhargaaaaaaaBhargaaaaBharehBhardwarenyaBhardwareBharddiskBharapnBharapannnnnnnnBharapannnnnnnharapannnnnnnnnnnnB	harapannnB
harapanlahB	harapankuB
harapankanBharapanharapanBharapancumaBharaoanBharaanBhappierBhapenyaaBhapalanBhanyiranB	hanyasajaBhanyanyamanBhanyalahBhanyaaBhanyBhantaBhanpirBhannyaBhannasuiBhankanBhanifB	hanguslahBhangatttttttB
hangantkanBhanekkkBhandyBhandsoapB
handsleeveB	handshockBhandsfreenyaB
handsetnyaBhandponeBhandphonetiamBhandlingBhandgripBhandbagB
hancuuuuurBhancurrrBhanbodyBhanayaBhanasuiBhamsterBhamshamsBhampuraBhamoirB
hammocknyaBhamburinBhambalanBhamaB
haluuussssBhalusssBhaloooooBhaloooBhaloBhalloBhalamnBhaknyB?hajahgahajagajajhsjsjshshsjshshsgahahahgshshshshshshshjsjsjsjsjBhainanBhailBhaicoB	hahhhahahBhahhhaBhahhahaaBhahhahaBhahhaBhahayBhahalBAhahajejdjfkkfkfkdkdkdkdkdjjdjdjdjdjjfjfjfjfjfjjffjfjfjfjfjfjfkfkfB.hahajakakdhdhdhdbdkfkrirjsbsjfkfkfncncmckckcjfBhahahshshhshshshshshshshshBhahahhahahahahhahahahaBhahahhahahahahahahhahahaBhahahhahahaB	hahahhaaaBhahahhaaBhahahahhahahahahhahahahahB
hahahahhaaBhahahahahahahhahhaaaaaB4hahahahahahahahahahahahahahhhhahhahhahsvhvhsbhhahahaB8hahahahahahahahahahahahahahahahahahahahahahahahahahahahaB'hahahahahahahahahahahahahahahahahahahahBhahahahahahahahagahahagahahaBhahahahahahahaaBhahahahahahaB
hahahahaaaB	hahahahaaBhahahahB
hahahaaaaaBhahahaaaBhahahaaB
hahaahhahaBhahaaaaaBhahaaaaBhahaaBhaganyaBhafalanBhafalBhaeiBhaegaBhaeapanBhaduwBhaduuuhBhaduuhhhBhaduuhhB
haduhhhhhhBhaduhhhhBhaduhhBhadoooohB	hadohhhhhBhadirBhadilnyaBhadilBhadihhhBhadiahxBhadiahnyBhadewhBhadewB
hadeuuhhhhB	hadeuuhhhBhadeuhhB
hadehhhhhhBhadehhhBhadeeuuhhhhBhadeeuuhB	hadeehhhhBhadeehB
hadeeeehhhBhadeeeeBhaddeuhB	haddehhhhBhaddeehBhadawBhadBhackingBhabyaBhablumBhabisnyaBhabislahBhabiskanBhabiaB
habbemanusB
haannnyyaaBhaahahBhaahaaBhaahB
haaddeehhhBhaaaaaaaddddddeeeeeeeBhaaaaaBhaaaaBgzlBgzBgyBgxtipuBgxgxncjccjfyfjchcyyfufuxysyuffyBgwedeeeBgwedeBgwaBguyssBguuuuuudBgutulBgutttttttttttttttttttttttttttttBgusssBgununganyarBgunknnBgunkanB	guncanganB
gunakannyaBgunakakB	gunaannyaBgunaaakakakakakakakkaakskkakBgumpalBgumakanBgumBgulungBgulingBgulaliBgukgukBgukBguidenyaBguideBgugusBgugukBguguguguguguguhuhuhuhuhuhuhBguguBguglingBguehBguedeBgudhikanBgudhhhtggggggB#guddddddddddsddddddssssssssshahhsbsBgudddB	gudangnyaBguardB
guandossssBguananyaBgtwBgtuuuuuuuuuuuuuuuuuuuuuuuuuuuuBgtuhBgtuanBgtoBgthuBgtauBgtaBgstteBgskBgsjsgsjsgsjsgshBgshsjkakavcdBgshockBgsgsgsBgsdgsdgB	groundnyaB	groundingB
grosirankuB
grisirankuBgrimisBgrgrBgretongBgrennBgrenjengB	grendlnyaBgregetanBgreeB'greatttttttttttttttttttttttttttttttttttBgratisssssssssssBgratissB	gratisnyaBgratisnyB
graphicnyaBgraphicBgransiBgrandongBgramnyaBgraedBgradBgrabnyaBgrabinstantBgqfgkyskstnBgpsmapBgpppBgppalahBgpapalahBgpBgoyanggBgoyahBgowesBgovlokB
gosokannyaB	gosendnyaB
gorillapodBgoriBgorenganBgoplakBgopekB4goooooooooooooooooooooooooooooooooooooooooooooooooodB,goooooooooooooooooooooooooooooooooooooooooddB+gooooooooooooooooooooooooooooooooooooooooodB(goooooooooooooooooooooooooooooooooooooodB&goooooooooooooooooooooooooooooooooooodB3gooooooooooooooooooooooooooooooooooodddddddddddddddB"goooooooooooooooooooooooooooooooodB%goooooooooooooooooooooooooooooooiooodBgoooooooooooooooooooooooooooodB:goooooooooooooooooooooooooddddddddfffffffffcddffddddfdffffB#gooooooooooooooooooooooooodddddddddBgooooooooooooooooooooooooodB"goooooooooooooooooooooooodddsdddddBgooooooooooooooooooodddddddddddBgoooooooooooooooooddddddddddddBgooooooooooooooooodB"goooooooooooooooodddddddddddddddddBgooooooooooooooooddddddddddddddBgoooooooooooooooddddBgooooooooooooooodBgooooooooooooodB"gooooooooooodddddgooooooooooodddddB"goooooooooooddddddddddddddddddddddBgoooooooooooddddddddddBgooooooooooiiiioooddddddddddddB goooooooooodddddddddddddddddddddBgooooooooooddddBgooooooooodddddBgoooooooooddBgooooooooddddddddddBgoooooooodddddddBgoooooooggggfffgooooogooogoBgoooooooddddddBgooooooddddddgooooooddddddBgooooooddddddddddddddddddddBgooooooddddddddddddB
gooooodlahBgooooodddddjahhsjsbsjsksjshsjsiB
goooooddddB	gooooodddBgoooododooddddB1gooooddddddddddddddddddddddddddddddddddddddddddddB
goooodddddB	gooooddddB
goooddddddBgoodssepatuB
goodsellerBgoodrecommendedBgoodproductB	goodpriceBgoodjoobB	goodiebagBgoodgxB(goodgahajajanshhshsjajajjajshshshhahahahB$goodddddddddddddddddddddddfdddddddddB!gooddddddddddddddddddddddddffffffBOgooddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddB.goodddddddddddddddddddddddddddddddddddddddddddB*goodddddddddddddddddddddddddddddddddddddddB(goodddddddddddddddddddddddddddddddddddddB'gooddddddddddddddddddddddddddddddddddddB"goodddddddddddddddddddddddddddddddBgooddddddddddddddddddddddBgoodddddddddddddddddBgoodddddddddddddddBgooddddddddddBgooddddddddB	gooddddddB"goodbaguspackingrapikalumayanbesarB	gondossssBgondolB	goncanganBgomiBgombrongBgolputBgolonganBgoldnyaBgoldenBgolaliB	gokilllllBgokillBgokiiilBgokeknyaBgojapBgoidBgoibBgogleB
gogglesnyaBgodjoobBgoddjobB;godddddddddddddddddddddddddddddddddddddddddddddddddddddddddBgoddddddBgodddddBgodddBgocenganB	gobloknyaBgoblogBgobBgntunganBgntinyBgndozBgnbrBgmyangBgmpngBgmpangBgmnlhBgmnhBgmnaneBgmbrnyBgmbarrBgmbargambarBgmailnyaBgmabarBglueBglownyaBglossnyaBglossBglogloBgliterBgldBglasssesBglamorBgknBgkkgBgkkanBgkkBgkatkanB	giwangnyaBgiwangB
gituuuuuuuBgituuuuBgituuuBgituuBgituqwetybsBgituinBgitudehBgituanBgitcyuBgirlyBgirlBgirimBginimahBginihBgimnaaaaaaaaaaaaaaBgimanacaranyaBgimanaaaBgimaBgillaaB	gilingnyaBgiiilllllaaaBgigitinBgigitanBgiftnyaB	gibranoziBgibaenBgiBghostingBghostBghjjkkgfddghjiydsadfhnnjiBghjjjbffgyuhhhgyuujhhgyB%ghjjhmmokgcssghnhbbvvffdhhhmmmlpotddxBghjhB2ghihvuutfhiitvbkonvfrgvvajsnznzjnzbzbzisksmblsllabB"ghfsfjklgdsadgjkkkvdswsfhjkgdeerggBghaibBghaBghBggytBggvjBggtBggpBggoooodddddddddBggoodBggodBggjmmBggjjjjkkkkkkkkkkkkkklkkkkkkkkkBgggtttttttttttttttBggggooooddddB'gggggggooooooooooooooooooooooooodddddddBggggggggggggrggggggggB6gggggggggggggggggggggoooooooooooooooooooddddddddddddddB
ggggggggggBgggggffffffffffBOgggggcggfddxvhuhvfdhtchhgggxxxfbcxcxghhbvccchhjjbcffghjbcxxcbjhbnbcxcnjhhcvnvhuBgggggaaaannnnnBggdfggggggghhhhhggffffggggghhhBgfdhbdjyBgeusBgeulisBgetetBgetahBgesuaiB
gesturenyaBgesturB	gespernyaBgesetBgeserxBgeryataBgeryBgerrynyaBgerryBgerobakBgercepBgerbangBgeratisB
gerakannyaBgerakanBgerahhBgeorgeB5geoheohdogdogdogdogdovdovfodvovdodgobdodhogdodgodgodgB	geografisBgeofanyBgenuineBgentongB
gentengnyaBgenkBgenitBgengsB	generatorBgeneralBgendosBgendogBgendaalaBgenapBgemstoneB	gemsstoneBgemsB
gemologistB
gemessssssBgemessBgemesinB	gemerisikBgembrotB	gemborkanB	gemboknyaBgembokBgemanaBgemB	gelombangBgeloB	gelisahknBgeligaBgelepBgelatinBgelapanBgelanyaBgelambirBgejalaBgeiwnheBgehBgeguguB	geeerrrrrBgedurBgedheBgedenyBgedegnyaBgedanBgeboyBgeblekBgebegBgearBgdvgviewibiBgdeanB	gddddddddB	gcojvigcgBgbtBgbisaBgbarBgayanyaBgaweBgatsbyBgatotBgatewayB	gaterlaluBgasukaBgasamaBgasalahB	garudanyaBgarudaBgarskinBgarrisBgarnetBgarmediaBgarisnyaBgarganyagarganyaBgardinerB	garbarnyaBgarapkanB
garapannyaBgarapanBgaramnyaBgapuasBgaptekBgappaBgaperluBgapapapaaaaaaB
gapapablahBgapakeBgapaBganzBganyampeBgantungnBgantunganyaBgantixBgantisssBgantikanBgantiiiBgantibateraiB	ganthanksBganterimakasihB
gantengnyaB	gantelanyBgantarBganokB#gannnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnB"gannnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnBgannnnnnnnnnnnnnnnnnnnBgannnnnnnnnnnnnnnnnnnBgannnnnnnnnnnnnnnnBgannnnnnnnnnnnnnBgannnnnnnnnnnnnBganjossB
ganfotonyaBganfosB gandozzzzzzzzzzzzzzzzzztttttttttBgandozzzBgandosssssssssssssssssssssssBgandosssssssssssssBgandossssssssssssBgandossssssBgandooosssssBgandooossssB
gandooosssBgandhozzzzzzzzBgancangBganbateBganasBgamrinBgampngBgamneyaBgamisnyBgamesirBgamesBgamersBgameboyBgambrnyaBgamblBgambaryaB$gambarrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrBgambarrrrrrrrrrrrrrrrrrrrrBgambarrrrrrrrrrrrrrrrrrBgambarrrrrrrrrrrrrrrrrBgambarrrrrrrrrrrrrrrrBgambarrrrrrrrrrrBgambarrrrrrrrrrB	gambarrrrBgambarnyB
gambarnnyaB	gambardanBgambaraBgambangBgambaarrB
gambaaaaarB	gamasalahB	gamantappB
gamabarnyaBgalicinBgalerryBgalatamaBgalanyaBgalakanBgaksBgakkkBgakgakBgakdiBgakayaBgakanBgakadBgajlasBgajelassBgajeBgajahnyBgajahBgajBgaisBgainBgahsjdidiidBgahshshshshshshshsBgagusB5gagisgisgisigsigsigsgishishihsihsohksishihskbdkbdkbkdBgagaraBgagangxBgagalinBgagajskskhjakaksnxbzkanabskzkabBgaessBgaeessB	gaeeessssBgadunganBgaduhBgadisBgadikasiB
gadiikirimBgadBgabusnyaBgabolehBgabocorBgabikinBgabesnyaB	gabakalanBgaapapaBgaannnnnBgaannBgaalakuBgaakBgaaannnnBgaaanBgaaaannBgaaaanBgaaaaanBgaaaaaaannnnnnnnnnB	gaaaaaaanB gaaaaaaaaaaaaaaaaaaaaaaaaaaaaaanB)gaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBgaaaBfykBfycBfyBfuuulllB
futuristikBfunsiBfungsionalitasnyaB	fungsikanBfungsiiiBfungiBfundoB	fullstarsB
fullscreenBfullhdB
fullbrightBfujifilmBfugsuBfuckBfuBftsalBftonyaBftonaBftocopyannyaB	ftocopyanBftftBfspBfsgsghedBWfsfdsfsssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssBfruityBfruitaminnyaB	fruitaminBfrogBfriskyBfreqBfrekwensinyaB	frekuensiBfreetB
freeongkirBfreeongBfrankyB	framerateBframenyaB	fragranceBfpglcB%fototpigppalahhhlumayanmksihhyahhmbaaBfotoooooBfotooBfotonyBfotokopiannyaBfotoinB
fotographyB	fotografiB
fotograferB
fotocopianBfotanyaBfositifBfosilBforoB	formatnyaB	formalnyaBformBforceBfootwearBfootoBfontnyaBfongBfompetBfoldBfokoknyaBfoilBfogBfocusnyaBfoangBflugBflowerBflowBfletchingnyaBfleshdealnyB	fleshdealBfleechBfleceeBflazhBflavourBflastB
flashsaleeB
flashlightBflashingBflashhhhhhhhB
flashflashB	flashdillBflashdelB	flashdaleBflasdealnyaBflanelB
flafournyaBfladhBfkfjfjgBfixingBfixelBfixateBfiuhhBfitsBfitnessBfitnahBfitingB	fisilitasB	fisiiknyaB
fisheyenyaBfishBfirstBfiringBfipakaiBfinyetBfinisingBfinishinBfinggerBfingerstyleBfingerprintnyaBfingerboardBfindBfinallyBfilnyaBfilenyBfilanyaBfikirainBfigureBfifaBfictureBficBfiberBfhjjkkljgfdsadghBfhjhfB	fhdhkkjhgBfgsixBfgnnBfgnBfghfcjgcchxBfggjlljjjjosssdddBffujkufddggjBffhB*ffffffffffffffffffffffffffffffffffffffffffBfffBfewBfeshopBfesBferomB
ferivikasiBferariBferancisBfendiBfelcroBfekingBfeatureBfdkdbsBfdgghjBfddsxsasdddfBfcBfatalBfastresponsB
fastcahgerBfastanaBfasstB
fassstttttBfasrBfasionB	fasilitasBfashsaleBfashaBfarisBfariasiBfantanyaBfamiliarBfambarBfamatexBfalamBfakturBfakkBfakBfairingBfailBfaidzinBfaeB
fadhilshopBfadbackBfactBfacingBfacebookB	fabrikasiBeyelinernyaBeyebrowBexternalnyaB	extensionBexspresBexspeyetBexsperietnyaB	exspedisiB
exspectasiBexspayetBexprsssB	expresssaBexpnyaBexpnyBexpiryB	expirenyaBexpidisiBexperiaB	expentasiB
expedisinyBexpdsBexpdBexpaidBexkulBexfireBexercitationBexerciseBexeB	exclusiveBexcilentBewBeveryB	evercrossBeverBeventBevaluasiBeuuuuyBeuuiiiiBetudeBetniknyaBetikatBetikadBetcBetalasiB
etalasenyaBetB
estiminasiBestimatenyaBestimateB
estimasinyBestimasikanBestetikaBessenyaB	essentialBessennyaBesrimasiBespedisinyaBespedisiBesoknyaB	esokannyaBeskulinBesensialBescapeBesananBerrowBerrotB	errorrrrrBerrornyaBerotBerorrBerimksiB	ergonomisBereredddeeeeeeeeeBerakanB	eracupnyaBequalizernyaBequalisernyaBepikkBepercosBepatanB
eoemekekkeB	enyakkkkkBenyakBentryBentangBenskBenqkBenoughBenjoyBenimBenhancernyaBengkBenggkBengganBengetBengapBengaBengB	energizerBenergiBenekBendutB
endullllllB	endoscopeBendosBendessBendakBencokBenaklahBenakkkkkkkkkkkkkkkkkkllB.enakkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkB
enakkkkkkkB	enakkkkkkBenakkkkkBenakkkkBenakkkBenakeunBenakenakBenakakkkkkkkkkBenajBenagaBenaaaakkkkkB	enaaaaakkB#enaaaaaaaaaaaaaaakkkkkkkkkkkkkkkkkkBenaBemukBempukkB	emosionalBemoryBemogaBemnkBemiB	emergencyBembosnyaBembanBembakBemasnyaBemangxBemangnyaBemanBemailnyaBemailkanBeluBelsaBelokBelitBelementBelemenBelektronikitB	elegannyaBelegandBeleganceBelectricnyaBelectricBelcoBelasticBelapBelahBelaganBelBekukBekstrimB	eksternalB
ekstensionB
eksspetasiBeksptasiBekspressBekspreesBekspiredBekspetasinyaBekspetasinsayaBekspektasilahBekspektasiiiB	ekspektasBekspekstasiBekspekB	ekspatasiBekslusifBeksekusiB	ekpentasiBekpedisiatauBekoooooBekooooBekooBekonimisBekhBekatraBekBejwjiwneoemwBeiusmodBeimBeiimmB!ehsjamsvsjamagsjsnsgsjsnnsjshhsjsBehoBehmBehhhhhBehhhB
ehhegehhegBeheheBegyptB	egronomisBegoBegkBegasBefsBefisienBefexBeferkrosBefelnyaBefekrifBefekifB9eerrrreeeeeeeeeeeeeeeeennnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnBeehhhBeeeeokeeeeeeeeeBeeeehB"eeeeeereeeeeeeeeeeeeeeeeeeeeeeeeenB-eeeeeeeeeeffffffffffffggggggggggggghhhhhhhhbbBeeeeeeeeeeeeehB+eeeeeeeeeeeeeeeereeeeeeeeeeeeeeeeeeeeeeeeeeB2eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBeeeeeeeeeeeeeeeBedyanBedunggBedunBedsBedodBedmBeditnyaBeditedBeditanBedinBedgjifseghiigfefhuvfhjjghhggedBedgeBederBedanB	ecpectasiBeconomisBecoBechonyasehariBechonyaBeceranBecerBebgoeyikwkgwisnsvuejdidoudBebelBeasyenglishBeasierB
earphonenyBearnyaB
earmuldnyaBearbudsBearbudBearbaBeapihhBeakBdzikrikaBdzBdynamaxBdyahBdwuBdwsBdwnkBdwhBdvBduuuhBduuuchhBduuBdutaBdusuruhBdustakanBdustaBdussBdusnyB
dusndusnyaB
dusbooknyaBdusbookBdusbokBdurableBduprosesB
duperbaikiBduperBdupasangkahBduoraBdununganBdunnBdunggBdungdungBdunganB	dumurahinBdummyBdumiBduluuuuBduluuuBdululahBdulayarBdulBdukunganBdukuhBdukuBdukanyaBduittttBduittB
duitnyapasBduiteB	dugunakanBdugemBdugantiBdugaankuB	duetalaseBduetBduehB	dueehhhhhBduduknyaBdudukinBdudhdnBduckBdubleBdubadanBduaribuBduapainBduanyBdualcoreBdualBdtungguBdtrmaB	dtrackingBdtokoBdtnyaBdtngnyBdtmptBdtmpatBdtkoBdtikBdtgnyaaaaaaaaaB	dtentukanBdtempatBdtelitiBdtekenBdtegBdtaroBdtanyaBdtangkapB	dtanggungB
dtanggapinB	dtanggapiBdtBdsukaBdsruhBdsrnyaBdsolBdsniiiBdsmpnBdslaluBdskripsiBdskpsiBdsiniiBdshB	dsekripsiBdsbB
dsayangkanBdsawahB	dsarankanBdsarBdryB	drumahnyaB	drskripsiBdrpadaBdrpB	drosipnyaBdrosipBdroshipBdropsitBdropshipperanB	dropshipeBdropahipBdroopshiperBdroopBdrofBdrngBdrmnBdripshipperrrrBdrillingBdribleBdribelBdrhBdrgBdresponBdresnyaBdresBdratnyaBdrastisBdrakBdraBdputarB	dptnyabyaBdprosesB	dprbaikinBdprBdpotoinBdpngnBdpkainyaBdpkaiBdpkaeB
dpindahkanBdpilihB	dphatikanBdpetnyaB	dperbaikiBdpenjelasannyaBdpengirimanBdpeaanB
dpasangnyaBdpasangB	dpakainyaBdoyokBdoyanBdowerBdoveBdouraBdoubtB	doubletipBdotonyaBdotBdorongBdorBdoppBdopeBdoorB
doooooooooBdoongBdonpetBdonlowBdonkerB	donglenyaBdongketBdongggggggggggggBdonggBdongelBdonakBdonaBdompetnyBdomoetB	dominanyaBdolterBdolpinBdoloreBdolorBdoloeBdoloBdollarBdokuuB
dokumennyaBdokonyaBdokoBdogantiBdogBdofBdoesnBdoesBdocumentBdocknyaBdocBdoanyaBdoantBdoankkkBdoanggBdoanganBdoaBdnulasBdniBdngerBdneganBdndhdBdmsBdmnaBdmasukanBdmasihBdluanBdlmyaBdlmnyBdlmanyaBdlebhnBdlapakBdkurirBdkttBdksihBdksiBdkrmkanBdkkBdkitttBdkitinB	dkirimnyaBdkiirimBdkiB	dkfigivkrBdketBdkcBdkadihBdjwbBdjvfjvgjvfhvcgBdjossB(djknahbdknabbbfjckdosjsjaknbsjzkababgwioB	djkdjrjroBdjiwoooB#djiwaaaaaaaaaaaaaaaaaaaaaaaaaaaaaksBdjiwaaaBdjieBdjiangBdjfgBdjeirjdjB3djdkkdkskadfldlslldldkfkskalfllhldlskdkdkdlfkfkdldkB
djdjdjsjshB2djdjdbhdhdhdhdhdhhdbdhduduhdhdhdhdhdhudhdhdhdhdhhdBdjdgabsjskdndbsnsnsnsndBdjayaBdjarumBdjanuonlineBdjajalBdizoomB	diwilayahB
diwakilkanBdiwadahiBdivstrapBdivobaB	divariasiBdivacumBdiungkapkanBdiunboxBdiulurBdiulasanB	diulanginB	diukurnyaBditwrkanBditvBdituntutBditunjukkanBditunjukBditundaB	ditulisinBditukrB
ditukarkahBdituanBditrmaBditrmB	ditranferB
ditrackingB	ditotalinBditotalBditontonBditonjolkanBditoleransiBditolakBditmptBditmbhBditlpBdititipBditiruBditirimaBditingkatknBditindakBditiBditesterBditestedBditessBditespunBditerimsB	diterimanB	diterimaiBditerimaaaaaaaaaaaaaaaaB	diterimaaB	diteriimaBditeriamBditerangkanB	ditepatinBditengahB	ditempqatB
ditempelinB	ditempeliB
ditelusuriBditekukBditekenBditeeimaB	ditebelinBditebakBditeBditawarBditaroBditapakBditanyaiB	ditanggalB	ditangangB
ditangaaanBditancapkanBditanamB	ditambahiBditaikinBditahanB	ditaburinBditB
disyangkanBdiswipeBdisusunBdisukainBdisukaiB
disuguhkanBdisudutBdistributorBdistreplessBdistorsinyaBdistorsiBdistanceB
displaynyaBdisorotBdisolasiBdisolBdisobekBdisoB	disktipsiBdiskripsinnyaB
diskripsiiB	diskripaiBdiskripBdiskrepsinyaBdiskinB	disketkahBdisitaBdisinikerenBdisiniiiiiiBdisinidisiniBdisimpulkanBdisimiBdisikatB	disiapkanB	dishooromBdishB	disetrikaBdisetnyaBdisetiapB	disetatusBdisetBdisesuaiBdiserutBdiseriBdisepelekanB
disepeleinBdisentuhB
disenterinB
disenenginBdisendalBdisemuaB	disemprotB
disematkanB	diselotipB	diseleksiBdiselBdisekelilingBdiseduhB	disebutinB	disebelahB
disebagianB
disebabkanB
discoutnyaB
discountyaBdiscounBdisconnectedB	disconectBdiscauntnyaBdisatuinBdisaringB
disanggupiBdisampulBdisampingnyaB
disamperinBdisalahgunakanBdisakuinB	disajikanBdisainBdisaBdirusakB	dirumahanBdirombakBdirinciBdirimnyaBdirikuBdirevisiBdireviewBdireturnBdirestaB
direspoundB
diresponyaB	direspondBdiresetBdirerimaBdirepackBdirendamB
diremehkanBdirekturBdirekomendedBdirekomendasikannBdirekomendasikamBdirekamBdiregistrasiBdirecBdireadBdirawatBdirautB	dirapihinB	dirangkaiB	diragujanBdipwetanyaanBdiputerBdipuringBdipunyaBdipundakBdipuasinBdiptongBdipsenB	diprosessB
dipromokanBdiprlerbaikiBdiprioritaskanB	dipreviewB	diprcepatBdipotooB	dipotonyaBdipostBdiposkanBdipolesBdipojokBdipnctBdipitBdipintaBdipinggirnyaBdipinggiranyaBdipinggirannyaB
dipilihnyaB	dipictnyaBdipickBdipesnanB
dipesannyaB
dipesankanBdipesaaaaannnnBdipermukaanB
dipermudahBdipermakB
diperlukanB
diperlebarBdiperhatiknBdiperhatiakanB
diperhalusBdipergelanganB	diperesarBdipercyaB
diperbaruiB	diperapihBdiperagakanBdipensiunkanBdipengrimanBdipengirimannyaBdipengenBdipengajianB	dipendingBdipemberitahuanBdipelajarinyaB
dipelajariB	dipekenyaBdipeehatikanBdipcB
dipassaranB	dipasasngB
dipasanginBdipantauB	dipangkalB	dipandangBdipandanB	dipanahinBdipalikB	dipaksainBdipakiaB	dipaketinBdipaketB	dipakenyoB
dipakenyaaBdipakenyB
dipakeknyaBdipakeeeeenyaBdipakainnyaB
dipakailahBdipakaiiiiiiiiiiiiiiiiiiiiiBdipakaiiBdipakBdipajeBdipahaBdipaduBdipackingnyaBdiotorisasiBdiotakBdiosengBdiorrrdeeeerrrrrrBdioptimalkanBdioprekBdioperBdionlineBdionkanBdiongkosBdiolshopBdioesenBdiodaBdiobrasBdiobgkirB
dinualakanBdinoticeBdinoteB	dinikmatiB
diniginkanB
dingunakanB	dinginkanBdinggoBdingdongBdindingBdinantiB	dinamonyaB	dinamakanB	dinaitkanB	dinaikkanBdinBdimulaiBdimukakuB
dimudahkanBdimobilB
dimiringinBdimintaiB	dimintaaaBdimilikBdimentokBdimencetBdimemoryB
dimelarkanBdimedanBdimauuBdimataB
dimasukkinBdimanipulasiBdimamaBdimalamBdimaksimalkanB
dimakluminBdimaeninBdilupainBdilotionnyaBdilombaBdilokasiBdiliveryBdilipatanyaB	dilintingBdilimpahkanBdilihtB	diliatnyaB
diletakkanBdilengkapinBdilenganBdilemparBdilemaBdilebihB	dilebarinBdilcdB	dilayaninBdilarangBdilanjutBdilalahB
dilainkaliBdilainB	dilabelinBdikuwalitasBdikutubBdikutiB
dikurirnyaBdikupingBdikupasBdikumpulkanBdikulkasBdikrmkanBdikripsiB
dikresekinB
dikrenakanBdikraB	dikotakinBdikonfBdikomunikasikanB	dikomplenBdikompaBdikomentarinBdikomentariBdikomenBdikolomBdikitttB	dikisaranB
dikiromnyaBdikirimmB	dikirimknBdikirimdiklarifikasiBdikirimalahB	dikiranyaB	dikiraiinBdikikBdikhususkanBdikhawatirkanBdiketutBdiketikBdiketiakBdiketernganBdiketerangannyaB
diketawainB	diketatinBdikerubutinB
dikerjakanB	dikerjainB
dikerdusinB
dikeraskanBdikenalBdikenaiBdikembanginBdikembaliknB	dikemasanB	dikelupasBdikeluarkannyaBdikeluarkaniBdikeluarkanB
dikeluargaB
dikelasnyaBdikeepBdikediriBdikategorikanBdikatainBdikataBdikaskusBdikasikB
dikasihkanBdikasihhBdikasibB
dikarnakanBdikardusnyaBdikaosnyB	dikantungBdikantongnyaB	dikambungBdikamarBdikaltimBdikalibrasiBdikaliBdikalaBdikakiiBdikakidikakiBdikadihBdikaciB	dikacanyaBdikabupatenB
dikabelnyaBdikabelB	dikabarinBdikaaiBdikaBdijwbbbBdijuslBdijulBdijntBdijminBdijitBdijgaBdijerseyBdijemurBdijemputBdijawaBdijatibeningBdijariB	dijangkauBdijakselB	dijaitnyaB	dijahitanBdijadiinBdijBdiitungB	diisapnyaBdiinstalkanBdiinstalasikanBdiinstalBdiinputBdiinjekB
diinhinkanB	diinginknBdiingatBdiinfoinBdiindfinkanB
diindahkanBdiincarBdiikutB	diiklasinB	diiklaninBdiikanBdiiBdihubungkanBdihubB	dihormatiBdihlgunakanBdihitungnyaB
dihitamkanB
dihilanginBdihhhB
dihentikanBdihembuskanB	diharapknBdihandalkanB	dihalusinBdihBdiguntingnyaBdigunkanBdigunanaBdigunakannyaBdigunakannhaB	digunaiimBdigulungBdigudangBdigoyangBdigorengBdigntiBdigilaiBdigigiBdigeserBdigemborB	digantungBdiganmabBdiganjelBdigambarrrrB	digambarrB
digambaarrBdigamabrBdifuserBdiftoB	difotonyaB
diflasdealBdifisikkkkkkkkkkkB
difikirkanBdifikirBdifhotoBdifakeqBdiestimasikanBdierimaBdiengarBdienakinBdiempangBdiekspedisiBdiduniaBdidukungBdidompetnyaB
didomesticBdidkirinB	didiskusiBdidiskripsiBdidinginkanBdidieuBdidieminBdideskriptionBdidesignBdidepanBdidengernyaB	dideketinBdidapetBdidalemB
didalammyaBdidaerahBdidadaBdicucuBdicubaBdicounternyaBdicoretBdicopyBdicontohB	dicongkelBdicolekBdicobaanBdicobaaaaaaBdicobaaaBdicintaiBdicicipBdichattB
dicetaknyaBdicetakBdiceritaainB	dicepetinBdicelanaBdicekkanBdicbaBdicbBdicazBdicatatBdicariinBdicantungkanBdicantelBdicancelB	dicairkanBdibyrBdibwahB	dibustnyaBdibunkusBdibungkusnyaBdibukakBdibukaaaBdibugkusBdibubleBdibubbleB	dibuatkanBdibpesanBdiboxnyaBdibotolBdiborongBdibordirBdiborB	diboonginBdibogorBdibodyBdibodohiBdiblzBdiblokirBdiblesBdiblasBdibjualB	dibioskopBdibinginkanBdibilasiBdibilagBdibidangnyaBdibicarakanBdibgmbarBdibgianB
dibesarkanB	diberkahiBdiberitahukanB
diberitahuBdiberiknBdibenrinBdibengkokkanBdibenakBdibenahiBdibelinB	dibelikanBdibeliiBdibelakanginB	dibedaainB
dibeberapaB
dibebankanB
dibeacukaiBdibcobabakanBdibbrpBdibazelB
dibayarkanB	dibayarinB
dibayangknB
dibayanginBdibarangnyaB	dibantingB	dibandungBdibandinginBdibandenganBdibandarB	dibalikanB
dibaleslahBdibajuBdibahasB
dibaguskanB	dibagusinBdibagBdibadolBdibadenB	dibadankuB	dibacanyaB	diawalnyaBdiaturdiaturBdiatassBdiasalBdiareaB	diarahkanBdiapusBdiaplikasiinB
diaplikasiBdiapkaBdiapitBdiapakaiaalaahBdiapainB
dianternyaB	dianternyBdiantenaB
dianjurkanBdianeB	diandelinB
diandalkanBdiancamBdiamterBdiamplasBdiamondBdiakuiB	diaktifknBdiaksihBdiainiBdiadukBdiaduinB	diaampingBdhuluB
dhsuauushsBdhrapknBdhngsBdhhhjcsbhtbfhBdhhhhhhhhhhhhhhhhhBdhdujsjeBdhdhdhdBdhdbsusB	dhasesuaiB	dharapkanBdhaniBdgunainBdgtBdgmbrBdgedeinBdgantiBdganbarB-dfjtdjdjsykugcgstdjgfyegjfudywudgkvcjvjcxhdydBdfixiditxiditBdfhbBFdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfdfBdeyB	dewhhhhhhBdewekBdewaBdevinaBdevBdeuinyaBdeuhBdettolBdetileB	detectionBdetecBdetaiB	destinasiB	destimasiB
desskripsiBdesmberBdeslBdesktopBdesktipsB	desksipsiBdeskrpsinyaB	deskrpisiBdeskrpB	deskropsiBdeskrisiBdeskripsinyaaaBdeskripsinyBdeskripsiinBdeskripsiiiiiiiiiiiiiiiiiiiiiiiB
deskripsiiB	deskripisB	deskripdiB	deskriosiB
deskrifsiyB	deskrifsiBdeskriapinyaBdeskrfsiBdeskresiBdeskrepsinyaB	deskrepsiBdeskpsiBdeskisiB	deskirpsiB
deskeripsiBdeskBdesisBdescriptionsBdescreptionBdescpBdesainyaBdesaBderjatBderajatBdepkrisiBdepkripsinyaBdepdepBdepanyaBdentalBdenimnyaBdeniBdenhamBdengwnB
dengungnyaB	dengerinnBdengasBdenganrBdengannpesananB	denganganBdengakBdengaanBdendruftB	dendenganBdendanBdendaBdenagnBdemurahBdemogaBdemageBdeluxeB	deliveryiBdeliveriB	deliciousBdeleteBdelcellBdelapanB
dekskripsiBdekrpsiBdekripsinyaBdekriosiB	dekkripsiBdekillBdeketanBdejganBdehoBdehmBdehhhhhhhhhhhhhhhhhhjjjjjBMdehhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhBdehhhhhhhhhhhhhhhhhhhhBdehhhhhhhhhhhhhhhhB	dehhhhhhhBdehgBdehbisaBdeghhBdeghB
degannnnnnBdefuncBdefinisiBdeepB	deehhhhhhBdeehhhhhBdeehhhhBdeeehhBdeeehBdeeeehhhB
deeeeeeeehBdeeeeeeeeeeehhhhhBdeeeeeeeeeeeeeeeeehBdeeechBdeeeBdeeBdedenganBdedekBdedBdeclareBdeckBdechhhhhBdechhhhBdechhBdechdechBdebfBdebesBdebbdndenncennxnxnnnBdebangBdeanganBdealnhaB	deadpixelBdeadBdeacriptionBddngerinBddlmBddlemntaBddipakekBddetelBddenganBddehBddddddddddddddddddddBddddBdcrBdcobainBdcesBdcepetBdcbaBdbyarkanB-dbxbxnxbxxnxnxnxnnxxbxbxbxnxxmxmxnxnnxnxnxnnxBdbutuhinB
dbukalapakBdbuatB dbsbdbdbdgxghshdhdhdhsjshshhshshBdbndingBdblzBdblasBdbilangBdbgnBdbeliBdbdnBdbdbdBdbayarB
dbayangkanBdbawahBdbawaBdbalasBdbagianBdbadanBdayatBdayanyaBdayangBdawchBdavidaBdaurBdaunyaBdatsunBdatngnyaBdatipadaBdatgBdateungBdatengyaBdatengnyBdatenBdatelineB
datatngnyaBdatatB	datascripBdatangyaB
datangnyaaBdatangnyBdatangnaB
datanganyaBdatamgBdatagBdatadataBdasternyBdasrnyaBdasinyaBdaruratBdarknyaB	darkbrownBdaripdaBdarimanaBdariapdaBdareBdaratanB	darangnyaBdarangBdaraiBdarahnyaB
dapetinnyaBdapayBdapaBdapBdaoetBdaoatBdanvtidakbterlihatBdantypeBdantaaBdantaB	dansesuaiB	danresponBdankurirBdaniBdangdutBdanganBdandrufB	danbtepatBdanauBdanapunBdampeB	dampainyaBdamiBdamgdutBdaluangB	dalemanyaBdaleBdalanyaBdalamnyBdalammyaB	dalamannyBdalahjsakksjdewaBdakwahBdakotaBdakadaB
dahsyatnyaBdahsyatBdahiBdahhhhhhhhhhhhhhhhhhhhhhhBdahhhhhhhhhhhhhBdahhhhhhhhhhBdahdahBdaguBdaganyaB
daganganyaB
daganganmuBdagB	daerahnyaBdaeiBdadatangBdadaaaBdachhhBdabestBdaatengBdaachhBdaaahhhhhhhB	daaahhhhhBdaaahhhBdaaahBdaaaghhhhhhBdaaaaanBdaaaaahBdaaaaaahhhhBdaaaaaahBdaaaBcyinBcyBcxBcweBcvfdaaBcuyyyyyyyyyyyyBcuyyyBcuyyBcuuyBcuuuyyyyBcuussBcuttonbdB	cuttinganB	cutingnyaBcuteeeeeeeeeeeeeeeeeeeeeeeB
cutbraynyaBcutbrayB cutbraaaaaaaaaaaaaaaaaaaaaaaaayyBcustrBcustonerBcustonBcustomesBcustomeBcustomBcustkuBcustemerBcussssB	cushionnyBcushionBcurveBcurlBcurirB	curhatnyaBcurcolBcurahBcupanBcunaBcumsnBcumsBcumbaBcumannnBcumahBcumaannBcumaaaBculupBculunBculasBcukurxB&cukuppppppppppppppppppppppppppppppppppBcukuppBcukupnnyamanB	cukuplaahB
cukupbagusB
cukupanlahBcukpBcukaBcuilikkkBcuilikBcuihB-cufucjvihihivuyfychvjvkbjvufuuvibobobohigivuvBcuepetBcucukuBcucoooooBcucoookBcucoooBcucookkBcucooBcucokkkkkkkkkkBcucikkkkkkjkjmnmnnnnnnnnnBcuccoooookkkBcuccoB	cuccccokkBcubaBcuatomerBcttnBctrdBctpB?ctdsryhsjsnjsksknzhzuuxhxgyxyzgzgyzyshhwoqpskjxhsbexfysjsnnqjjnBctatanBcsoneBcsizeBcrystalBcryptonBcryBcrxBcrunchyBcruelityB
crosscheckBcrossBcroscekBcrootBcroootB	crlananyaB
crlanaanakBcritaB	crepesnyaBcremyBcreatineBcreamxBcrankBcrakBcrahBcrackingBcrBcpgBcpettBcpekBcpaapBcpBcozyBcoyyyyyyyyyyyyyyyyyyyyB	coyyyyyyyBcoyyyBcoxokBcowonyaBcowonyBcowoknyaBcowokkkBcovokBcoverageBcourrierBcourirBcourierB
counternyaBcouldntB	costumizeBcostumBcostomerBcostemBcossBcosnyaBcosmeyikB
correspondB
correctionB	corongnyaBcoretBcorelBcorasBcoraknyaBcorBcopottBcopotnyaBcopianBcooyyyBcooxkBcooperativeBcooollBcooolBcoooBconyBconverBcontrastBcontrasBcontohinBcontekanB	consequatBconsecteturBconsBconnnectB
connectnyaB
connectkanBconneckBconncetBconicalBcongekBconfrimBconfirmasinyaBconfidensialB
confermasiBconexBconetingBconenBconectornyaB	conectionBcomprehensiveB	componentBcompoB	completedBcompletBcomplenB	complaintBcomplaiBcompetitiveB	compeleteBcompangBcompBcommonBcommodoB	comentlahBcomeentBcombedBcolumbiaBcoloursBcolonganBcolokkannyaBcolokkanB	colokankeBcollorB
collectionB
colesterolBcolekB	colectionBcoksuBcokltBcoklatttB	coklarnyaBcokekBcokaltBcokBcoilnyaBcoiB	coffeenyaBcofeeBcodBcocooooooookkkBcocolahBcocoksB	cocoklahhBcocokkkkkkkkkkkkkkBcocokkkkBcocokkkBcocokkanBcocojBcocoBcockBcociksBcocikBcobsBcobekBcobaqBcobanyBcobalahBcobakBcobaiinBcobaaaaaaaaaaaaaaaaaaaaaaaaaaaaBcobaaaaaBcobaaaaBcobaaaBcoaxialBcoatBcoakBcntumkanBcmplainBcmlnBcmkanBcmiiwBcmgBclutchesBcloudBclouchBclothingBclothBclosingBcloseBclorisBcloningannnnnBclnBclipperB	clipernyaBclikBcliiingBcliiiingBclearlyB	cleansingB
cleanfreshBcleanerBclaynyaBclassB
claritynyaBclamBckckccBckckB
ckakakakakBcjmanBciyyyBcivaBciusBcitranyaBciriocsiBciracasBcintaiBcingciripitB	cinderungBcincinyaBcincauuuuuuuuuuuuuuuuuuuuuuuuuuB	cincailahBcincaiBcinBcimaBcilokanBcilikkkkBcilikBcilanBcikupBcikidapBcikenBcikeasBciiiiiiiiaaaaaamiiiikkkkkkkkkkBciiihuuuyyyyyyyyyyyyyB	cihuyyyyhBcihuyyyBcihuyBcihuiiiiiiiiiiiBcihuiiiicihuiiiiBcicncinBcicinnyaBciciBcibinongB
cibangkongBciamikkkkkkB	chxhccjdxBchtnyaBchstnyaBchrgerBchragerBchosiyahBchosesBcholatosnyaBchoiceB
chitatonyaBchinesseB	chicdenimBchhatBchffhfBchesterBchestBcheryyBcheryBchelseaBchekoutBchekingB
cheesecakeBcheerluxBcheaperBchdufjsgfjdjcufuxBchayB
chattinganBchatsBchatlB	chatinganBchatbyaBchatanBcharisaB	chargeranBchargB	characterBcharBchangerB	chanelnyaBchanB
chalcedonyBchakepBchairBchainaB	chagernyaBchagerBchagBcgdaBcgbvcdfvbbvxsdggfdcbjhdddvbBcgbcvfbBcgBceypatBcewenyaBceweknyaBcewekkkBcewaBcewBcevetBcetokBcethaBcetaknyaB
cetakannyaBcerminBcermatBceriwisB	ceritaainBceriaBcerewetBcerajBcerahhhBcerahhBcerBceptttttttttttBcepttBcepstB!ceppppppppppppattttttttttttttttttBcepopppppppppeeeeeeettttttttBcepetyB'cepetttttttttttttttttttttttttttttttttttB
cepettttttB	cepetttttBcepetttBcepetinBcepetbangetBcepekBcepeetttBcepeetBcepeeetBcepeatBcepeanBcepcopBcepayBcepavtBcepatyaBcepatttttttBcepatthanksBcepatssBcepatrecommendedBcepatlahBcepatinBcepatereaponyaBcepatbarangB(cepatbabababaabananaanmamtapsesuaigambarBcepatanBcepataBceparltBcepakBcepaattttttttttttBcepaaattBcepaaaaatttB
cepaaaaaatBcentreBcentangBcengkehBcenelB	cencinnyaBcenahBcempremgB
cemplunginB	cemeranyaBcelupkanBceleretannyaBcelengannyaBcelenaBcelebuuuuuuuuBcelciusBcelannyaB
celananyahBcelananyBcelanaaBcelanBcelahBcelabaB	celaanayaBcelaBcekresiBcekingBcekatanBcecuaiBcebatB
cebanantapBcebananBceapatBcealanaBcdnganBcdkBccvyhikkBcctvnyaaBcctvnyaBcctBccpetBccepatBccabutBcbangBcbakBcaurBcattBcatoknyaB
catokannyaBcathokanBcatetanBcaterpillarB	catcitcotBcatchyB
catatannyaBcatataBcastamerBcasshanBcasioBcashingBcashbacknyaB	cashannyaBcashanBcasanyaBcasannyaBcarrierBcarinyaBcargonyaBcargernyBcardioBcardinBcardigannyaBcardiganBcaranyahBcaranxBcarangBcaraaaBcaptainBcapsulBcapsgelB	cappucinoBcaplokBcapedehB	canvasnyaBcantumBcantolannyaBcantikkkkkkkkBcantiiikB	cantiiiikB	canteekkkBcantecantelanB	cangkolanB	cangkokanBcangihBcancleB	cancellerB
cancelilngBcanBcampaiBcampBcamersBcamdigBcalvinBcallbackBcaliperBcalanaBcakupanBcakeupBcakeppppwarrB-cakepppppppdahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhBcakeppppppoopB
cakeppppppB	cakepppppBcakepanBcakeepppBcakeeppBcakeepB
cakeeeppppB	cakeeepppBcakeeeppBcakeeepBcakeeeeppppB!cakeeeeerreeeeeeeeeeeeerrrrrrerepB	cakeeeeepB#cakeeeeeeeeeeeeeeeeeeeeeeppppppppppBcakebBcakapBcairkanB	cahayanyaBcagavsvsjsjsjsjsisnbsiskslalaBcadngnBcadeBcadangnB
cadanganyaBcadBcactBcacingBcachB
cacatcacatB	caberawitBcabelnyaBcabaiBcaasBbzaB	byeeeeeeeBbyariB bxndjkemsmndmdmfjfjfjvhahshhhjhhBbwsBbwrBbwneranBbwlumBbwianspBbwhnyBbwhannyaBbwgusBbwahanBbwBbvvBbvsdxghntybtdvfcgjnhbvdcfsdBbvhhBbvdfvBbvdffBbuzzingBbuuuuuaaaagusssssBbuuuuaaanngggeeeettttB	buttonnyaBbuttonBbutiqueBbutikBbusterBbussssssBbussetttBbusnyaBbusettBbuseeetB	busananyaBburutBburungngnyaBburukkkkBburuangBbursaBburrBburnBburesBburekBburberryB
buooosssssBbunyunyaBbunusnyaBbunusBbuntukBbunkusanBbungsuB	bungkusssBbungkussB	bungkusnyBbundlingnyaBbundlingBbundleBbundaBbunciiittttBbunchemBbunchBbunBbulumBbulshitBbulpenBbullshitBbulelengBbulannnnBbulangBbukusBbuktiinbuktiinBbukqnBbuklpkBbukitB	bukapalakBbukapakBbukangB	bukalqpakB
bukalappakB
bukalapnyaBbukalapkB	bukalapalB#bukalapakkkkkkkkkkkkkkkkkkkkkkkkkkkBbukalapakkkkkBbukalapaaaaakkkB	bukakapakBbukainB
bukadompetBbujukBbujednyaBbuildingBbugarBbuffetB	bufferingBbuesaarBbueningBbudimanBbudiB	budgetnyaBbudahBbubutkanBbubuknyaB
bubllepackBbublewarpnyaB	bublecrapBbubgkusBbubblesBbuayaBbuatvkauBbuattBbuatlahBbuatapaBbuasBbuarkanBbuanginBbuangettttttBbuangetsBbuangatBbuaatBbuaaanyaaakBbuaaangeeeettttBbuaaaanggggettBbuaaaangetttBbuaaaaangetttttttttBbtreynyaBbtrenyaBbtreiBbtreBbtmonoB	bterainyaBbteraiBbsnsnhsjsjssbsjskzkkzsmnsnsbssBbsndjkxkbbhsnzhsjnhhznBbsliB)bshshsytshejsjsgsgsgvshsjsnshebdhdhensjjsB9bshosnaoansoahshfjdjdidfhkdhruudueuehheidwbjskzbslsvdhcnsBbshisjsisbsBbsdaBbscsBbrushnyB	bruntusanB
brukatnnyaBbrukatB
brsngkutanBbrsiBbrsaingB	brsahabatBbrrubahBbrrroBbrrharapBbrotherBbrossB	brosernyaBbrooiBbrokenBbrokatBbrohBbrodolBbrocketBbrobroBbrngyaBbrngyBbrngxB
brngnyaaaaBbrnBbrmutuBbrlumBbrlapisB
brlanggannBbrlanggananBbrkrgBbrkmbngBbrklBbrjumlahBbrjlnBbrjayaBbringBbrikutnyBbrikanBbrigeaccBbrhrpBbrhasilBbrharapBbrgnyaluBbrgnyaaaaaaaBbrgnhaBbrgngBbrgmyaBbrgbungBbrenB
brekeleeeeBbrekanBbrecketBbreB
brdempetanBbrbisnisBbrbicaraB	brbelanjaBbratiBbratBbrarangBbrankasBbraniBbranhBbrangyaB
brangnyaaaBbrangnxBbrangkatBbrangkasBbrangdahBbrandnyaBbrandedBbranchBbrambangBbragBbradulB	bracelettBbrabgBbraangBbqnyakBbqiBbqgusBbproBbpkBbozzzBbozzBboxnyBbouskuBboubleBbotolnyBbothBbotB	bosternyaBbossssssssssssssssssseBbossssssssssssssssssBbosssssssssssssssssBbossssssssssssssBbosssssssssssBbosssssssssdB
bossssssssB!bosssssassasssaassasssssszzzzzzzsB	bossskuuuBbossskuuBbosskuuuBbosquuuBbosquuBbosquB
bospancingBboskuuuuuuuuuuuuBboskuuuuB4boskuhhhhjjjhhhjjjhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhjjhhBboskuhhhBbosenBbosannyaBbosaBborozBboronganBbormaBbordirrB
bordirnyaaB	bordirnyaBbordilanBbopoBbopelnyaBbooyaaahBboowwBbootnyaBbootlopBbootingB
boosterrrrB
boosssssssBboossBboosBbooosB
boooosssssBboooosssBboooossBboooooosssssBbonuzBbonuusBbonusyaBbonussBbonusnysBbonusnyakapanB	bonusnyahBbonusntaBbonusiBbonusbonusannyaBbonusanBbonnesBboniusBbonisBbongsorB	bongkaranBbongkahBbonekanyBboncosanBboncosBboncelBbonBbomnyaBbomberrBbolonginB$bollleeehhhhhhhhhhhhhhhyyyyyyyyaaaaaBbolehkahBbolehhhBboldBboksBbokoBbokinBbokehnyaBbokehBbokeehBbokBbojoB	bohonglahBbohongiBbohlamBbogusBboehBbodongBbodokBbodoBbodhBbocinBbocilB
boccoorrrrBbocahhBbobrokBbobovrBbobotBbobleBbobBboasaBboardingBboalakBbnyjBbnyBbntikBbnqttBbnljaB
bnjynhbgdfBbnibBbngusBbngungBbngttttttttttttttttrB	bngttttttBbnggtttttttttttttttttttBbnggttttBbnggtBbngettzB	bngetttttBbngettttbngettttttttttttttBbngetttBbngeeeetBbngeeeeedbngeeeeedBbngeeBbngBbnegtBbneerBbnaggetBbnagetBbnBblutoothB	bluthoethBblusukanBblusnyaBblusBblurnyaB	bluotoothBblunyaB
bluetutnyaBbluetotBbluetootnyaBbluetoohBbluethotnyaB
bluescreenBblubenBblsanBblowBbloutodBblouseBbloumBbloroBbloototB	blootoothB	bloothothBblolBbloetothnyaB	bloetoothBbloemBbloborBblnjakanBblnjaaBblnjBblngBblkBbljaBblitzBblintikBblindB	blezernyaBblewahBblepetanBblengB
blendernyaB
bleberrrrrB	blazernyaBblaxBblassBblasanB1blankkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkBblankkkBblakangBbladusBblacknyaBblablablablablablaBblaasBbkuBbknyBbklnBbklanBbkerjaBbkalapakBbkalanBbkadBbkBbjuneBbjukuB"bjuddjursvjiiijhfdfjjjytrdgiiuhhggB
bjngkusnyaBbjkmbdgiojgyukmvhkmbcfhmBbjkkjcvbkmnnjhvgBbjiBbjdirudjdjfdBbjarBbizBbiyayaBbiwBbitcolBbisuuuuBbissaBbissB
biskuitnyaBbisiuuuuuuuuBbisepBbisapakeBbisankomnekBbisabisaBbisaaaaaBbisaaaBbisaaBbiruanBbirosBbiriBbirelBbirdokB
birdfhidhdBbirdBbirBbipBbiorenyaBbioblitzBbioaquaBbintngB	bintiknyaBbintanyaBbintanhB	bintangnyB	binrltangBbininyaBbingutB	bingungmuBbingumgBbingkinBbingkaiB	bingitzzzBbingitzBbingitttBbingitsssssssssssB
bingiiitttBbingiiitBbingiiiitttzzBbingiiiiittttyyyyyBbingiiiiiitzzzzzBbingiiiiiiitttssssssBbingiiiiiiiittttB
bingiiidddBbinggowBbinggggoiiiiiiiiiiiittttBbinderBbindangB
binbintangBbinatangBbinasaBbinahongBbilpointBbilngnyaBbillqualitiBbilgnyaBbilangxBbilanBbikinyaB
bikinannyaBbikinanBbikimBbikiinBbikerBbijinyaB	bijaksanaBbijakkahBbijakBbiiiiaasaaaaaaaaaaBbihBbigsizeBbigpondBbigoBbiggerBbidaraBbidaBbicarabicaraBbicaraaaaaaaaaaaaaaaaaaaaaBbicaraaaaaaaaaaaaBbicaraaaaaaaaaaBbicaraaBbibleBbiayabkirimBbiatBbiasawB	biasannyaBbiasakanBbiasaaaaaaaaaaaaaaaaaaaaaaasaaB)biasaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaB!biasaaaaaaaaaaaaaaaaaaaaaaaaaaaaaB biasaaaaaaaaaaaaaaaaaaaaaaaaaaaaBbiasaaaaaaaaaaBbiasaaaaaaaaaBbiasaaaaaaaB
biasaaaaaaB	biasaaaaaBbiasaaBbiarknBbiaikBbiaarBbiaaaaasssaaaBbiaBbhnyaBbhneBbhknBBbhjsjjsjjmnnsjjsjsjjsjjsjjsjjsjjsjjshshjsjsjjsjsjjsjsjjsjsjjsjsjjsBbhhhjjjaiaoaoakakkabaBbhhahhahahhaBbhhBbhghB&bhgddfghyfdcvbnnnvvvcxbvccvbbvcccbvvvvBbhasaBbhanxBbhabBbgyBbguzBbguuusBbguusBbgussssssssssssssBbgusssssssssssssB	bgussssssBbgusssssBbgusnyBbguslahhhhhhhhhBbguslahB	bgusbagusBbgunBbgugganBbgugBbguBbgtvyaahBbgtulahBbgttttttyttyttttttBbgttttttttttttttttttBbgttttttttttttBbgtttttttttttBbgttttttttttBbgtttttttttbgttttttttttttttBbgtttttttttBbgttttttB bgtttbgttttttttttttttttttttttttrBbgtsBbgtlahBbgtfjamkbsjmakaBbgtenggaBbgtbgtbgtbgtbgtbgtbgtbgtbgtbgtBbgtbgtBbgslahBbghBbggtBbgdssBbgddddBbgddBbgctBbgannyaBbgaiB	beztberryBbezleBbeyondBbewoknyaBbewokanBbewokBbeutBbetuuuulBbetumpukBbetulllllllllllllllllllllllBbetulkanBbetulinBbetulahBbettBbetsBbetmenBbetkualitasBbeteleBbetapaBbesuBbestttttBbesttttBbesttB
bestsellerBbestlahBbesstBbessB
besokannyaBbesellerBbesegelBbesatBbesarsehinggaBbesarrrrrrrrrrrrrrrrrrrrrrrrBbesarrrBbesarinB
besarbesarBbesaaarBberwanaB	berurusanB
beruntusanBberulangkaliBberukatB	bersungsiB	berstikerB
berspungsiBbersinarBbersilatBbersikapBbersikanBbersiinBberseriBberseratB	bersepedaBbersentuhanBberseliweranBbersangkutanB
bersambungBbersaBberryBberqualitasBberpugsiB
berprofesiB
berpngaruhB	berpindahBberpaBberombakBberolahB	bernyanyiB	bermutuuuBberminguB	bermingguBbermerekB	bermaslahB	bermasalaBbermangfaatBbermanffaatBbermanfaaatBberlualitasBberlogoB	berlobangBberlipatB	berlimpahB	berlenganB	berlendirB
berlelebihBberlatihBberlarutBberlangaananBberlamaBberkwalutasBberkwalitetBberkuwalitasB	berkumpulBberkumandangBberkualitisBberkualitaskecewaBberkualitasberkualitasB
berkualitaB	berkreasiBberkomitmenBberkkuaaliitasBberkibarBberkhualitasBberkhasiaaaatBberkendaraanB
berkendaraB
berkembangBberkemahB	berkarpetBberkapurB	berkantorB
berkantongB
berkancingBberkahzaBberkacaBberjumpaBberjualanyaBberjualannyaBberjualanlahBberjualBberjlnB
berjerawatBberjapanBberjamurBberjamB	beritikadBberitaB
berinovasiB	beringnyaB	berimbangBberikutxBberikutbuktiBberikemajuanB"berhasilllllllllllllllllllllllllllBberhasiilllllllllllllllllllB
berhasiaatBberhamburanB	bergungsiBbergunaaB	bergubgsiB	bergliterBbergemaB	bergariisB	bergabungBberfungsiiiiiiiiiB
berfungsiiB
berfungsfiB	berfingsiBberfikirB
berfanfaatBberexpetasiBberetikaBberendamBberembunBberekspektasiBberefekBbereeeeeeeeesssssBberedarB
bereblanjaBbereakBberduriB
berdirinyaBberdietB
berdengingB	berdempetBberdayaBberdasarB
berdangangB	berdandanBberdagangnyaB	berdadangBbercoverBbercocokB	berceritaBbercandaBbercakapB	bercahayaBberbutarB
berburunyaBberbuahB	berbohongBberbobotBberbisnislahBberbiruB	berbintikB
berbintangB
berbincangBberbicaraberbicaraBberbicaraaaaaaaaaaaaaaaaaassssBberbicaraaaaaaaaaaaaaaaaaaBberbicaraaaaaaaaaaaaBberbicaraaaaaaaBberbicarBberbiacaBberberatBberbenjaBberbelanjanyaBberbayarBberbayanganB	berbayangB	berbayaarBberbarenganBberbalutB
berbahagiaBberbagiBberbadanBberawalBberattttB	beratnianBberasalBberaromaBberantaiB	beranikanBberanggapanBberamalB
beralaskanBberaktifitasBberaksiBberakB
beraahabatBbepumBbentulanBbentukxB
bentuknyaaBbentuknyBbentukanBbentangBbensinB
benrfungsiBbenrB
benningtonBbenjoBbeningtangkaiB	beningnyaBbenihnyaBbenibgBbengyaBbengnyaBbengkongBbenerrrrBbenerrrBbenerenBbeneerrrB	benchmarkBbenaranBbenahiBbenahBbenagBbenBbembeliBbelutB	belummmmmBbelummBbelsnjaBbelpasiBbelowBbelonBbeloBbelnjaBbelmuBbellomB
belkangnyaBbeljrlahBbelinyBbelikBbelieveB	beliebersBbelibisBbelibeetBbeliauBbelehBbeldenBbelasBbelannjaBbelanjjaBbelanjahB	belanjaanB
belanjaaaaBbelanggananBbelancaBbelakngBbelajaBbekrjaBbekiBbekasssBbekassB
bekasannyaB	bekalapakBbekBbejibunBbeibehBbehBbegituuuuuuuB	begitupunBbegitudicobaBbegeteB
begerinjulBbegayaanBbegahB
befungsisyB
beerjualanBbeenB
beemanfaatBbeekualitasBbeekerjaBbeeginiB	beefungsiBbedulBbedsBbedilerBbedhaBbedekBbedakanBbedahBbedaaaaaaaaaaaaaaaaaBbedaaaBbecomeBbeckdorB
becarefullBbecandaBbebonusBbeberpaaBbeberpaBbebeknyaBbebebBbebaskanBbebaniB	beautynyaBbeauBbeatBbearBbeangnyaBbeangBbeanBbeacoupBbduBbdheyherBbdeBbdankuBbcraBbcBbbqBbbossBbbjghjmjBbbiiiiaaassaaBbbgusBbbgtBbbgjkkbBbbcBbbbuuuaaangggeeetttsssBbbbbbbbkkkkkkkkkkkB,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbB	bbbbbbbbbBbbbbbbBbbansnsjsjskskskskskskdkdnsnsjsBbbahanBbbagusBbbabbababaaggggusBbazelBbaystoryBbayrnyaBbayikkBbayarnyaBbayarkanBbayarinBbayanginnyaBbayanginBbayangannyaB
bayangankuBbaxakBbawrangBbawanyaBbawalB	bawainnyaBbawahnyBbawagBbawaanyaBbawaaanBbavusBbaukBbaugusBbaufengBbatuxBbatuakikBbattryBbattreyBbattreryB	battrenyaBbattreB
battrainyaBbattraiB
batterynyaBbattereBbattBbatruBbatrreBbatrnyB	batreynyaBbatrenyaaaaaaaaaaaaBbatrentaBbatrangBbatraixBbatokBbatinB	batiknyaaB	baterynyaB	baterinyaBbatereyBbaterayBbaterBbateeBbatasanBbatanyaBbatangframeB
batangbsdhBbatangasBbatanganBbatalBbatagorBbataeraiBbasssBbassnyBbasisBbasiconBbasiclyBbasicBbasesusiB	baselayerBbasaahBbasaB'baruuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuBbaruuuBbarusBbarungBbarulahBbarsangBbarringB
barrangnyaB	barqngnyaBbaronB	barometerBbarokallahufiikB	barokahanBbarnyaBbarnmangnyaBbarngnyB
barnangnyaBbarnangB	barnagnyaB	barnagnahBbaringBbargainB	barengnyaB
barcodenyaBbarbieB
barbershppBbaratBbararangnnyaBbarapanBbaranvBbaransB	baranngyaBbarannfBbarankBbaranhxaB	baranhnyaBbaranhgBbarangyeBbarangvsesuaiBbarangsesuaiB	barangsdhBbarangsB	barangnyqBbarangnyasihBbarangnyanyaBbarangnyamahBbarangnyahbarangnyahBbarangnyaaaaaB	barangnuaB	barangnhaBbarangngnyaB	barangnayBbarangnBbaranggggggggggBbarangggggggB	baranggggBbarangdiB	barangcptB	barangbygBbarangbsesuaiBbarangbsangatB
barangbnyaBbarangbdipackingBbarangaBbaranagBbarakallohuBbarakallahuBbarakallaahB	barabgnyaBbarabfBbaraangnyaaaBbaraaangB	baraaaangB	bapriknyaBbapinBbaokB
baofengnyaBbanyarBbanyanBbanyajBbanyaakB	banyaaakkBbanyaaaakkkkkkkBbanyaaaaakkBbanyaaaaaaaaakBbanyaBbanyB	banwahnyaBbantuinB	bantuanyaB
banrangnyaBbanrangBbanqetttBbannyakBbankrutBbanknyaBbanknyBbanjarmasinBbanhetBbangwtjashhsnanajajakaBbangusssssssssssssssssssssssssBbangungBbangunanBbangunBbangueteBbangtzBbangttB	bangsaaatBbangsaaBbangoBbangnyaBbangheetB	bangguettBbanggatBbangeyBbangetzzB
bangettttzB!bangetttttttttttttttyytttttttttttB0bangetttttttttttttttttttttttttttttttttttttttttttB"bangetttttttttttttttttttttttttttttB bangetttttttttttttttttttttttttttBbangettttttttttttttttttttttttttBbangettttttttttttttttttttttBbangetttttttttttttttttttBbangettttttttttttttttttBbangettttttttttttttttBbangettttttttttttttBbangetttttttttttB bangetttbangetttytttttttttttttttBbangetssssssssssssssssBbangetssssssssssssBbangetssssssB
bangetokehB	bangetdehBbangegB
bangeettzzBbangeetttttBbangeettttszzzzBbangeetttbangeettttttB
bangeettssBbangeeetttttttttBbangeeetttttBbangeeeetttttB$bangeeeeetttttttttttttttttttttttttttBbangeeeeeetttybangeeeeeetttBbangeeeeeetttBbangeeeeeeeettttttB$bangeeeeeeeeetttttttttttttttttttttttBbangeeeeeeeeetB*bangeeeeeeeeeeeeeeeeeeeeeeeeetttttttttttttB	bangeddddBbangeddBbangattttttttttttttttBbangakBbangaaaaaatBbanetBbandungbandungB
bandrolnyaBbandarlampungBbandaraBbanciiBbanbetBbanarBbananaBbanaBbalsemB	balotellyBbalmaniBbalmB	ballponitB	ballpointBballpoinB	ballotelyBbalinyBbalintB	balingnyaB	balingdanB
balikinnyaBbalikanB
baliglobalB	balesnnyaB	balesanyaB
balesannyaBbalesanBbalekBbalasnyaBbalapakBbalanjaBbalancerBbalanceBbalaasBbalaBbakuleBbakulBbakuBbaksonyaBbaksoBbakiBbakeBbakatBbakasanBbakangBbajuyaBbajuyBbajuweBbajunyaaBbajunaBbajakBbajajBbaitBbaisaBbaingBbainBbaimBbaileyBbailBbaikyBbaikusuauauauauB
baiksemogaBbaikpBbaiknnyaBbaikmBbaikkkkkkkkkkkkkkkkkkBbaikkkkkkkkkkBbaikkkkkkkkB
baikkkkkkkB	baikkkkkkBbaikkkkBbaikjikaBbaikanBbaijBbaiikBbaiiiikkkkkBbaiiiikBbaiiBbaihBbaibaikBbaiBbahwqBbahussBbahusBbahunyaBbahqnnyaBbahaxB
bahasilnyaB	bahasanyaBbaharnyaBbaharBbahanyyaBbahanyeBbahannyaaaaB	bahannyaaBbahannuaB	bahannnyaBbahannhaB
bahanhalusBbahangB
bahanbbatuB"bahanbaguscepetsampaisesuaipesenanBbahanaBbahahahahahahaBbahabnBbahaBbagysbajanajbsjabajaBbagysBbaguwssBbaguuzB5baguuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuussssB'baguuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuusB#baguuuuuuuuuuuuuuuuuuuuuuuuuuuuiuusB baguuuuuuuuuuuuuuuuuuuuuuuuussssBbaguuuuuuuuuuuuuuuuuuuuuuusB baguuuuuuuuuuuuuuuuuuuuuusssssaaB#baguuuuuuuuuuuuuuuuuuusssssssssssssB!baguuuuuuuuuuuuuuuuuussssssssssssBbaguuuuuuuuuuuuuuuuussssssssB baguuuuuuuuuuuuuuuusssssssssssssBbaguuuuuuuuuuuuuuuusssssBbaguuuuuuuuuuuuuuusBbaguuuuuuuuuuuuuusssssssBbaguuuuuuuuuuusssssssssssssssB!baguuuuuuuuuuusssssssssaassssssssB"baguuuuuuuuuusssssssssssssssssssssB!baguuuuuuuuuussssssssssssssssssssBbaguuuuuuuuuussssB.baguuuuuuuuusssssssssssssssssssssssdddddddddddBbaguuuuuuuuusssssssssssssssBbaguuuuuuuusssBbaguuuuuuusssssssBbaguuuuuuusssBbaguuuuuusssdB
baguuuuuusBbaguuuuussssssssssssBbaguuuuussssssssssBbaguuuuussssssBbaguuuuussssBbaguuuuusssB#baguuuussssssssssssssssssssssssssssBbaguuuusssssssB	baguuuussB
baguuuusasB4baguuussssssssssssssssssssssssssssssssssssssssssssssB!baguuusssssssssssssssssssssssssssBbaguuussBbaguusssssszzzB%baguussssssssssssssssssssssssssssssssBbaguusssssssssssssssssssssBbaguussssssssssB
baguusssssB	baguuslahBbaguszBbagusvsesuaiBbagusussssssssssssssssBbagussssssszBbagusssssssszzzzzsssssssBbagussssssssswwwB$bagussssssssssssssssssssssssssssssswB3bagusssssssssssssssssssssssssssssssssssssssssssssssB2bagussssssssssssssssssssssssssssssssssssssssssssssB/bagusssssssssssssssssssssssssssssssssssssssssssB+bagusssssssssssssssssssssssssssssssssssssssB*bagussssssssssssssssssssssssssssssssssssssB)bagusssssssssssssssssssssssssssssssssssssB'bagusssssssssssssssssssssssssssssssssssB.bagussssssssssssssssssssssssssssssssddssssssssB$bagussssssssssssssssssssssssssssssssB#bagusssssssssssssssssssssssssssssssB"bagussssssssssssssssssssssssssssssB!bagusssssssssssssssssssssssssssssBcbagusssssssssssssssssssssssssssessssssssssssssssssssssssssssssssssssssssssssssssdddssssssssssssssssBbagusssssssssssssssssssssssssBbagussssssssssssssssssssssBbagusssssssssssssssssssdddddddBbagussssssssssssssssssBbagusssssssssssssssssBbagussssssssssddBbagusssssssssbagusssssssssBbagussssssssddBbagussssssddddBbagusssssbagussssssssssssB
bagussssddB!bagussssaasssssssssssssssssssssssBbagussslahhBbagussdBbagussbagusssssssBbagussaBbagusoB	bagusnsihB
bagusnmtpiBbagusnmamkakkakakkakakakakkakaBbaguslqhBbaguslhB
baguslahhhBbaguslahbaguslahB	baguslaahBbagusjiBbagusihBbagushshhshhahshshhhahshhBbagusgusgusBbagusganB
baguscumanBbaguschjnvfcBbagusbangetB.bagusbagussssssssssssbagusbagussssssssssssssssBbagusbagusbagusBbagusbagusanBbagusassBbagusaBbagulsBbagueB	bagubagusBbaguasanBbagsuB#bagsirkandnwliksopqbbajsndmsmsmmsmsBbagoosBbagoooooooooisssssBbagoesssssssssssssBbagoesssssssssssBbagnyaBbagiuuuuuusB(bagissssssssssssssssssssssssssssssssssssBbagisssssssssBbagisB	baginilahBbaginBbagimuaBbagikanB	bagiannyaBbagiaBbaghsB
bagguusdddB"baggusssssssssssssssssssssssssddddBbaggtBbagggusBbagetttBbagauzbagauzhsjsnnsnsjBbagarangnyaBbagandosBbafusBbafanbBbaeeBbaeahBbadyBbadusBbadrolB	badmintonBbadlahBbadanyaBbadanqBbadannBbadaiiiiB
badaibadaiBbadagBbadaaaaaaaaaaaiiiiiiBbadaBbacoonBbackpackingBbackdoorB	backcoverBbacinBbacanyaB	bacaannyaBbacaanBbabyblueBbabsgetBbabiiiBbabagusBbaarokallahBbaaraangBbaangnyaBbaangeetBbaanBbaaikBbaaiikkkBbaaiikkBbaahhhhhhhhBbaaguuussssBbaaggguuuussssssssssBbaaangeeetttBbaaaikB
baaaiiikkkBbaaaguuuuusssssseBbaaagguuuuuuuusssssssBbaaagguuuussssBbaaadaiiiiiBbaaaangeeeetBbaaaaguuuuuuuuuuuuuuusssssssssBbaaaaguuuuuuuussssssssBbaaaaguuuusssssBbaaaagggguuuuussssssssB1baaaagggggguuuuusssssseeeeekkkkkkaaaaaallllllleeeB baaaaarrrrraaaannnnggggnnnyyyaaaBbaaaaaaguuussssssssssssB)baaaaaaatrrrrrrrraaaaaanbnggggggnnyyyyaaaB#baaaaaaaggggggguuuuuussssssssssssssB'baaaaaaaannnnnnngggggggeeeeetttttttttttB)baaaaaaaaaaagguuuuuuuuuuuuuuuuuuuusssssssB-baaaaaaaaaaaaaguuuuuuuuuuuuuuuuuuuuuyssssssssB"baaaaaaaaaaaaaaaaagggggguuuuuussddBazzzzzzzzzzzzzzzzzBazusBazilB	aziiiplahBayyyBayyaayBayunanBayeBawwetBawsomeB$awowowwowowowowowowoowowwkwowkwowkwoBawlB	awetttttyBawetttawetttttBawettlahBawetsBawetnyaBawetanBawekBaweeetttttttttttttttttttB
aweeeettttB
aweeeeetttBaweeeeetBaweeeeeeeetBawedhBawanBawalnyeBawalnyBawBavengersBautorollB	automatixBautofokusnyaB	autofokusBautentikasiBautenticBauratBaukeyBaukBaujqkbeusmwsjBaufinvB	audzubikaBaudiBauBaturrrBatumBatuBatowBatletBatlantikBatkBatensiBataukahBatasanyaBatahBasyiqueBasyiikBasyiiiikB	asuuuuuuuBastimasiBastagaaBastBassmBassisiBassezB	assesorisBassemblyBassalmualaikumBaspekBaspeakerBaspalBasoyyyyBasoyBaslinyagoodB	aslinyaaaBaslinxBaslinnyaBaslinaB
asliiiiiiiB	asliiiiiiBasliiBaslihBasliasB	asliannyaBasinnyaBasilBasiklahBasikkkkBasikbbingitsB$asikasikasikasikasikasikasikasikasikBasiikkkkBasiiiiiiiiiiiiiiiiiiiiikkkkBasianB2ashshjshsnsndjdjdhsusegdhdjsbjsjsjsjdjdjdjeebdjendBasepnyaBaselinaB	aseeeeeekB.asdfghlkejehidiwjwbhsisiwjhehjejebwhiejejdnjddB/asdasdasdasdasdassdasdksoixndbeuuxjslosjebdbhddBasallBasalkanBasaleBasalannnnnnnnnnnnBaryBarsipBarsenalBarrrghBarmyyBarmynyaBarmoreBarlojiBarigatouBariBarepBarcterixBarcsaberBarcBarapanBarahnyaBaraBaquilaBaquascapenyaBaquariumBaquB%aqpesenygbigsizekokdikirimallsizeyaaaBaqnyaBapusBaproveB
appreciateBappnyaB	applikasiBappiiiikkkkBapparelBapotekBapoBapnB	aplikatifB
aplikasiyaBaplikasikannyaBaplikasiinnyaB
aplikasiinBaplifierBaplgBapknyaBapkikasinyaBapkahBapikkkkBaphBapesnyaBapelBapekkkkkBapeessBapeBapckingBapaxBaparhanBapapaBapanBapalgBapalahhBapalaguBapakhB
apajadinyaB	apabilaanBapaaaaaaaaaaaBapaaaBaosjBanywhyBanywayyBanyerB	anycastyaBanyanganBanyangBanyakanBanyBanwarBanugerahBantvrBantrBantmanBantislipB
antisipasiB	antingnyaBantikB	antihujanBantepBantenBantemnyaBantebbbbBantahBanrhBankkuBanjurBanjirrrBanjirrB	anjinggggBanjiirrBanjeliB	anjayyyyyBanjayyyyBanjayBanjaayB	anjaaaaayBanhankanBangunnnBangungBangryBangnesBangleBangingB	anggurnyaBanggurBanggungjawabkanBanggungjawabanBanggunanBanggrekBanggpBanggipBanggaranBangellBangelBanganBanelloBanehxBanecastBandroitBandoridBandbunBandaikanBancuurrB	ancurrrrrBancurrrrBancuranBanchorBanatomiBanarchyBanamBanakxBanakqBanaknyBanakkBanakanBanajzbaksnaksjsnnsnsBanahBanagbaganaanagbaganafnatBamwatBamuratelB
amssshyobtBamsongBamsBamrikBampyunnnnnnnB
ampuuunnnnBampuunBampunnnnnnnnnBampunnnBamplyB	amplitubeB	amplifierBampereBampekBampaoBamoledBamnBammiinBamkasihB
aminnnnnnnBamiiinnBamiiiinB	amiiiiiinBamgaBamethystnyaBamericaBamehBameB	amburadurBamblasBamblBambingBambimBambiguBambarB
ambalannyaBambalanBamazfitBamaxBamatiranBamatirBamannnnnBamannnnBamannnBamannBamanmantappppB	amankerenBamanamanBamanahamanahBamanaBamalBamakaliBamajingBamadBamaannnBamaanBalwaysBalvonsoBalvinBaluuusssB
alusssssssBalureBalumuniumnyaBalumniBalukalBaluhBaltitudeB	altimeterBalternativeBalsoBalshopBalrightBalphaBaloeBaloaiqykamzBalngkhBalmuniumBalmtBalmariBallisanurseryB	allhmdllhBallhasilBallhamdullilahBallhamdulillahBallhB	allahummaB
alkoholnyaBalkalineBalitaBaliquipBaliquaBalincoBalihBalifiaBalifBaliaresiBaliB	alhmdullhBalhmdulillhBalhmdulilahhBalhmdililahBalhamudlillahBalhamudlilahBalhamiBalhamdullahBalhamdulilqlahBalhamdulillhBalhamdulillahnyaBalhamdulillahirobbilB$alhamdulillahhhhhhhhhhhhhhhhhhhhhhhhBalhamdulillahflashsaleBalhamdulillaaahBalhamdulillaBalhamdulilkahB,alhamdulilahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhBalhamduldulillahBalhamduiillahBalhBalgkahBalfianBalexBalergiBalchamdulillahB
alcatelnyaBalayBalatblangsungBalasxBalassepatunyaBalasanyaBalankahBalangkhBalangBalamiinBalamdulillahBalahamduliaBalaamiinaaaaaB	alaamaaakBakyBakuuuuBakutuuBakuranBakuntBaktivitasnyaBaktivasiBaktingBaktifinBaksiB'akshdndismdjiskdosmsidkfjfdkfvnvhvhfhcjB
aksessorisBaksesoriBaksesB
akreditasiBakorrrrrrrrrrrrrrrrrrrrrrrBakomBakmjBakkuBakjshdjbdhxjxjdjdnjdjdkddBakiknyaBakhrnyBakhiyatiBakhirxBakhinyaBakhBakgBakazaBakasiaBakariBakapBakamBakakabspsnsbsnBakageirotoggpgmsjB&akaahekwllaajabhaajajajakakowehakajalaBakaBajyaBajsjhsbwkwalsnbzbaasmbzbssBajpBajojingB.ajnwusnvsunsbsujbbdjdjjsijshshjsijsbshsjsjhssjB	ajiplahhhBajiippBajiiippB	ajiiiibbbBajiiibbbbbbbbbbBajiiibBajiibbbBajiibbBajieebBajieblahBajiebBajiblahBajibeBajibbbbbbbbbnbbbbbBajibbbbbbbbB	ajibbbbbbBajibbBajehBajayaBajarkanBajarBajalanaBajajkanssammabsbbB
ajajahahajBajaibnyaBajahhhhhhhhhhhhhhhhhhhhhhhhhhB
ajabragnyaBajaajajajwgdhakdjdbhakskfBajaahBajaaaaajaaaaBajaaaaaahhhhhhhB+ajaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaassaaaaaaBajaaaaaaaaaaaaaaaaaaaaaaBajaaaaaaaaaaaBajaaaaBaitemBairsoftBaioBainnaBainmentBainBaimBaiiiiiiighhhhhBaidzinBaichunBaibonnyaBahyBahxBahunBahlinyaBahirnyaBahirBahihhiBahhhhhBahapwhsbsuwjsBahapBahanknB
ahankannyaBahankannBahahahhaBahaayyBahaBagussssBagusBagunaBagtBagreeBagrBagannnnnnnnnnnnnnnnnnnnnnnBaganeBagakkBagahBagacBagaakkBagaaBafterallBafrikaBafijsfnafbatjfanfajfbaBafiatBafghjvnBafganBaewtBaerphoneBaerBaepeBaemberBaekaliBaegiBaefBadzhanBadwmBaduuuhhBaduuuhBadulBadukBaduinBaduhhhhBaduhhBaduhayBadslBadriBadonanBadminyaBadmiBadlahB	adjustmenBadipisicingBadikuBadikkuBadiBadenmBademnyaBademmmmmBademmmademmmBadehhhBadeemnBadeeemmmBadeeeemBaddB	adaresponBadapunBadanyqBadanlahBadamB
adalapisanBadakanBadaanB adaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBadaaaaaaBadaaaaBadaaaBacungkanBacungiBacuhkanBactuallyB	activitasBactivasiBacsesorisnyaBacsaberBacsBackingBackhBacilBacihB
acessoriesBacesorisBacepBaccsBaccountB
accesoriesBacanBacakanBacaiBabsenBabonnyaBabonBablmBabjsssBabizzzzzzzzzzzzzzzzzzzzzzxxzxzBabizzzzzBabizzzzBabizssssB abisttttttttttttttttttttttttttttBabissssssssssssssssssssssBabisssssssssssssssssssssBabisssssssssssssBabisssssssssssBabisssssssssBabissssssssBabiissssB	abiiiisssBabiiiiisB5abidbgdhsbsvhabvsjndvzhjzvvzhgxvvxvhzgzgzggxgsgzgshshBabdroidBabdiB>abcdjdksoelsmsbsmdbsksbskdldmdndndldmdndnsnnsnskslsmdmdmcmdcmcB#abcderghijklmnopqrstuvwxyzabcderghiBabcdefghijlamnopqrstuvwxyzB abcdefghijklmnopqrstuvwxyzhhuyvvBabcdefghijklmnopqrstuvwxyzabcdBabcBabauBabangnyaBabaikanBabaiBabahBabadiBabadB	ababhanssBabaalBaayaB aaweeettttttttttaaweeettttttttttB
aavaliableBaatauBaasalBaasBaampingBaaminnB	aamiiiienBaamaBaamBaaliBaalamiinBaakBaajaaaaaaaaa
��
Const_5Const*
_output_shapes

:��*
dtype0	*��
value��B��	��"��                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '       '      !'      "'      #'      $'      %'      &'      ''      ('      )'      *'      +'      ,'      -'      .'      /'      0'      1'      2'      3'      4'      5'      6'      7'      8'      9'      :'      ;'      <'      ='      >'      ?'      @'      A'      B'      C'      D'      E'      F'      G'      H'      I'      J'      K'      L'      M'      N'      O'      P'      Q'      R'      S'      T'      U'      V'      W'      X'      Y'      Z'      ['      \'      ]'      ^'      _'      `'      a'      b'      c'      d'      e'      f'      g'      h'      i'      j'      k'      l'      m'      n'      o'      p'      q'      r'      s'      t'      u'      v'      w'      x'      y'      z'      {'      |'      }'      ~'      '      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'      �'       (      (      (      (      (      (      (      (      (      	(      
(      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (       (      !(      "(      #(      $(      %(      &(      '(      ((      )(      *(      +(      ,(      -(      .(      /(      0(      1(      2(      3(      4(      5(      6(      7(      8(      9(      :(      ;(      <(      =(      >(      ?(      @(      A(      B(      C(      D(      E(      F(      G(      H(      I(      J(      K(      L(      M(      N(      O(      P(      Q(      R(      S(      T(      U(      V(      W(      X(      Y(      Z(      [(      \(      ](      ^(      _(      `(      a(      b(      c(      d(      e(      f(      g(      h(      i(      j(      k(      l(      m(      n(      o(      p(      q(      r(      s(      t(      u(      v(      w(      x(      y(      z(      {(      |(      }(      ~(      (      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(      �(       )      )      )      )      )      )      )      )      )      	)      
)      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )       )      !)      ")      #)      $)      %)      &)      ')      ()      ))      *)      +)      ,)      -)      .)      /)      0)      1)      2)      3)      4)      5)      6)      7)      8)      9)      :)      ;)      <)      =)      >)      ?)      @)      A)      B)      C)      D)      E)      F)      G)      H)      I)      J)      K)      L)      M)      N)      O)      P)      Q)      R)      S)      T)      U)      V)      W)      X)      Y)      Z)      [)      \)      ])      ^)      _)      `)      a)      b)      c)      d)      e)      f)      g)      h)      i)      j)      k)      l)      m)      n)      o)      p)      q)      r)      s)      t)      u)      v)      w)      x)      y)      z)      {)      |)      })      ~)      )      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)      �)       *      *      *      *      *      *      *      *      *      	*      
*      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *       *      !*      "*      #*      $*      %*      &*      '*      (*      )*      **      +*      ,*      -*      .*      /*      0*      1*      2*      3*      4*      5*      6*      7*      8*      9*      :*      ;*      <*      =*      >*      ?*      @*      A*      B*      C*      D*      E*      F*      G*      H*      I*      J*      K*      L*      M*      N*      O*      P*      Q*      R*      S*      T*      U*      V*      W*      X*      Y*      Z*      [*      \*      ]*      ^*      _*      `*      a*      b*      c*      d*      e*      f*      g*      h*      i*      j*      k*      l*      m*      n*      o*      p*      q*      r*      s*      t*      u*      v*      w*      x*      y*      z*      {*      |*      }*      ~*      *      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*      �*       +      +      +      +      +      +      +      +      +      	+      
+      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +       +      !+      "+      #+      $+      %+      &+      '+      (+      )+      *+      ++      ,+      -+      .+      /+      0+      1+      2+      3+      4+      5+      6+      7+      8+      9+      :+      ;+      <+      =+      >+      ?+      @+      A+      B+      C+      D+      E+      F+      G+      H+      I+      J+      K+      L+      M+      N+      O+      P+      Q+      R+      S+      T+      U+      V+      W+      X+      Y+      Z+      [+      \+      ]+      ^+      _+      `+      a+      b+      c+      d+      e+      f+      g+      h+      i+      j+      k+      l+      m+      n+      o+      p+      q+      r+      s+      t+      u+      v+      w+      x+      y+      z+      {+      |+      }+      ~+      +      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+      �+       ,      ,      ,      ,      ,      ,      ,      ,      ,      	,      
,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,       ,      !,      ",      #,      $,      %,      &,      ',      (,      ),      *,      +,      ,,      -,      .,      /,      0,      1,      2,      3,      4,      5,      6,      7,      8,      9,      :,      ;,      <,      =,      >,      ?,      @,      A,      B,      C,      D,      E,      F,      G,      H,      I,      J,      K,      L,      M,      N,      O,      P,      Q,      R,      S,      T,      U,      V,      W,      X,      Y,      Z,      [,      \,      ],      ^,      _,      `,      a,      b,      c,      d,      e,      f,      g,      h,      i,      j,      k,      l,      m,      n,      o,      p,      q,      r,      s,      t,      u,      v,      w,      x,      y,      z,      {,      |,      },      ~,      ,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,      �,       -      -      -      -      -      -      -      -      -      	-      
-      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -       -      !-      "-      #-      $-      %-      &-      '-      (-      )-      *-      +-      ,-      --      .-      /-      0-      1-      2-      3-      4-      5-      6-      7-      8-      9-      :-      ;-      <-      =-      >-      ?-      @-      A-      B-      C-      D-      E-      F-      G-      H-      I-      J-      K-      L-      M-      N-      O-      P-      Q-      R-      S-      T-      U-      V-      W-      X-      Y-      Z-      [-      \-      ]-      ^-      _-      `-      a-      b-      c-      d-      e-      f-      g-      h-      i-      j-      k-      l-      m-      n-      o-      p-      q-      r-      s-      t-      u-      v-      w-      x-      y-      z-      {-      |-      }-      ~-      -      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-      �-       .      .      .      .      .      .      .      .      .      	.      
.      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .       .      !.      ".      #.      $.      %.      &.      '.      (.      ).      *.      +.      ,.      -.      ..      /.      0.      1.      2.      3.      4.      5.      6.      7.      8.      9.      :.      ;.      <.      =.      >.      ?.      @.      A.      B.      C.      D.      E.      F.      G.      H.      I.      J.      K.      L.      M.      N.      O.      P.      Q.      R.      S.      T.      U.      V.      W.      X.      Y.      Z.      [.      \.      ].      ^.      _.      `.      a.      b.      c.      d.      e.      f.      g.      h.      i.      j.      k.      l.      m.      n.      o.      p.      q.      r.      s.      t.      u.      v.      w.      x.      y.      z.      {.      |.      }.      ~.      .      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.      �.       /      /      /      /      /      /      /      /      /      	/      
/      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /       /      !/      "/      #/      $/      %/      &/      '/      (/      )/      */      +/      ,/      -/      ./      //      0/      1/      2/      3/      4/      5/      6/      7/      8/      9/      :/      ;/      </      =/      >/      ?/      @/      A/      B/      C/      D/      E/      F/      G/      H/      I/      J/      K/      L/      M/      N/      O/      P/      Q/      R/      S/      T/      U/      V/      W/      X/      Y/      Z/      [/      \/      ]/      ^/      _/      `/      a/      b/      c/      d/      e/      f/      g/      h/      i/      j/      k/      l/      m/      n/      o/      p/      q/      r/      s/      t/      u/      v/      w/      x/      y/      z/      {/      |/      }/      ~/      /      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/      �/       0      0      0      0      0      0      0      0      0      	0      
0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0       0      !0      "0      #0      $0      %0      &0      '0      (0      )0      *0      +0      ,0      -0      .0      /0      00      10      20      30      40      50      60      70      80      90      :0      ;0      <0      =0      >0      ?0      @0      A0      B0      C0      D0      E0      F0      G0      H0      I0      J0      K0      L0      M0      N0      O0      P0      Q0      R0      S0      T0      U0      V0      W0      X0      Y0      Z0      [0      \0      ]0      ^0      _0      `0      a0      b0      c0      d0      e0      f0      g0      h0      i0      j0      k0      l0      m0      n0      o0      p0      q0      r0      s0      t0      u0      v0      w0      x0      y0      z0      {0      |0      }0      ~0      0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0      �0       1      1      1      1      1      1      1      1      1      	1      
1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1       1      !1      "1      #1      $1      %1      &1      '1      (1      )1      *1      +1      ,1      -1      .1      /1      01      11      21      31      41      51      61      71      81      91      :1      ;1      <1      =1      >1      ?1      @1      A1      B1      C1      D1      E1      F1      G1      H1      I1      J1      K1      L1      M1      N1      O1      P1      Q1      R1      S1      T1      U1      V1      W1      X1      Y1      Z1      [1      \1      ]1      ^1      _1      `1      a1      b1      c1      d1      e1      f1      g1      h1      i1      j1      k1      l1      m1      n1      o1      p1      q1      r1      s1      t1      u1      v1      w1      x1      y1      z1      {1      |1      }1      ~1      1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1      �1       2      2      2      2      2      2      2      2      2      	2      
2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2       2      !2      "2      #2      $2      %2      &2      '2      (2      )2      *2      +2      ,2      -2      .2      /2      02      12      22      32      42      52      62      72      82      92      :2      ;2      <2      =2      >2      ?2      @2      A2      B2      C2      D2      E2      F2      G2      H2      I2      J2      K2      L2      M2      N2      O2      P2      Q2      R2      S2      T2      U2      V2      W2      X2      Y2      Z2      [2      \2      ]2      ^2      _2      `2      a2      b2      c2      d2      e2      f2      g2      h2      i2      j2      k2      l2      m2      n2      o2      p2      q2      r2      s2      t2      u2      v2      w2      x2      y2      z2      {2      |2      }2      ~2      2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2      �2       3      3      3      3      3      3      3      3      3      	3      
3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3       3      !3      "3      #3      $3      %3      &3      '3      (3      )3      *3      +3      ,3      -3      .3      /3      03      13      23      33      43      53      63      73      83      93      :3      ;3      <3      =3      >3      ?3      @3      A3      B3      C3      D3      E3      F3      G3      H3      I3      J3      K3      L3      M3      N3      O3      P3      Q3      R3      S3      T3      U3      V3      W3      X3      Y3      Z3      [3      \3      ]3      ^3      _3      `3      a3      b3      c3      d3      e3      f3      g3      h3      i3      j3      k3      l3      m3      n3      o3      p3      q3      r3      s3      t3      u3      v3      w3      x3      y3      z3      {3      |3      }3      ~3      3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3      �3       4      4      4      4      4      4      4      4      4      	4      
4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4       4      !4      "4      #4      $4      %4      &4      '4      (4      )4      *4      +4      ,4      -4      .4      /4      04      14      24      34      44      54      64      74      84      94      :4      ;4      <4      =4      >4      ?4      @4      A4      B4      C4      D4      E4      F4      G4      H4      I4      J4      K4      L4      M4      N4      O4      P4      Q4      R4      S4      T4      U4      V4      W4      X4      Y4      Z4      [4      \4      ]4      ^4      _4      `4      a4      b4      c4      d4      e4      f4      g4      h4      i4      j4      k4      l4      m4      n4      o4      p4      q4      r4      s4      t4      u4      v4      w4      x4      y4      z4      {4      |4      }4      ~4      4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4      �4       5      5      5      5      5      5      5      5      5      	5      
5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5       5      !5      "5      #5      $5      %5      &5      '5      (5      )5      *5      +5      ,5      -5      .5      /5      05      15      25      35      45      55      65      75      85      95      :5      ;5      <5      =5      >5      ?5      @5      A5      B5      C5      D5      E5      F5      G5      H5      I5      J5      K5      L5      M5      N5      O5      P5      Q5      R5      S5      T5      U5      V5      W5      X5      Y5      Z5      [5      \5      ]5      ^5      _5      `5      a5      b5      c5      d5      e5      f5      g5      h5      i5      j5      k5      l5      m5      n5      o5      p5      q5      r5      s5      t5      u5      v5      w5      x5      y5      z5      {5      |5      }5      ~5      5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5      �5       6      6      6      6      6      6      6      6      6      	6      
6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6       6      !6      "6      #6      $6      %6      &6      '6      (6      )6      *6      +6      ,6      -6      .6      /6      06      16      26      36      46      56      66      76      86      96      :6      ;6      <6      =6      >6      ?6      @6      A6      B6      C6      D6      E6      F6      G6      H6      I6      J6      K6      L6      M6      N6      O6      P6      Q6      R6      S6      T6      U6      V6      W6      X6      Y6      Z6      [6      \6      ]6      ^6      _6      `6      a6      b6      c6      d6      e6      f6      g6      h6      i6      j6      k6      l6      m6      n6      o6      p6      q6      r6      s6      t6      u6      v6      w6      x6      y6      z6      {6      |6      }6      ~6      6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6      �6       7      7      7      7      7      7      7      7      7      	7      
7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7       7      !7      "7      #7      $7      %7      &7      '7      (7      )7      *7      +7      ,7      -7      .7      /7      07      17      27      37      47      57      67      77      87      97      :7      ;7      <7      =7      >7      ?7      @7      A7      B7      C7      D7      E7      F7      G7      H7      I7      J7      K7      L7      M7      N7      O7      P7      Q7      R7      S7      T7      U7      V7      W7      X7      Y7      Z7      [7      \7      ]7      ^7      _7      `7      a7      b7      c7      d7      e7      f7      g7      h7      i7      j7      k7      l7      m7      n7      o7      p7      q7      r7      s7      t7      u7      v7      w7      x7      y7      z7      {7      |7      }7      ~7      7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7      �7       8      8      8      8      8      8      8      8      8      	8      
8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8       8      !8      "8      #8      $8      %8      &8      '8      (8      )8      *8      +8      ,8      -8      .8      /8      08      18      28      38      48      58      68      78      88      98      :8      ;8      <8      =8      >8      ?8      @8      A8      B8      C8      D8      E8      F8      G8      H8      I8      J8      K8      L8      M8      N8      O8      P8      Q8      R8      S8      T8      U8      V8      W8      X8      Y8      Z8      [8      \8      ]8      ^8      _8      `8      a8      b8      c8      d8      e8      f8      g8      h8      i8      j8      k8      l8      m8      n8      o8      p8      q8      r8      s8      t8      u8      v8      w8      x8      y8      z8      {8      |8      }8      ~8      8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8      �8       9      9      9      9      9      9      9      9      9      	9      
9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9       9      !9      "9      #9      $9      %9      &9      '9      (9      )9      *9      +9      ,9      -9      .9      /9      09      19      29      39      49      59      69      79      89      99      :9      ;9      <9      =9      >9      ?9      @9      A9      B9      C9      D9      E9      F9      G9      H9      I9      J9      K9      L9      M9      N9      O9      P9      Q9      R9      S9      T9      U9      V9      W9      X9      Y9      Z9      [9      \9      ]9      ^9      _9      `9      a9      b9      c9      d9      e9      f9      g9      h9      i9      j9      k9      l9      m9      n9      o9      p9      q9      r9      s9      t9      u9      v9      w9      x9      y9      z9      {9      |9      }9      ~9      9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9      �9       :      :      :      :      :      :      :      :      :      	:      
:      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :       :      !:      ":      #:      $:      %:      &:      ':      (:      ):      *:      +:      ,:      -:      .:      /:      0:      1:      2:      3:      4:      5:      6:      7:      8:      9:      ::      ;:      <:      =:      >:      ?:      @:      A:      B:      C:      D:      E:      F:      G:      H:      I:      J:      K:      L:      M:      N:      O:      P:      Q:      R:      S:      T:      U:      V:      W:      X:      Y:      Z:      [:      \:      ]:      ^:      _:      `:      a:      b:      c:      d:      e:      f:      g:      h:      i:      j:      k:      l:      m:      n:      o:      p:      q:      r:      s:      t:      u:      v:      w:      x:      y:      z:      {:      |:      }:      ~:      :      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:      �:       ;      ;      ;      ;      ;      ;      ;      ;      ;      	;      
;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;       ;      !;      ";      #;      $;      %;      &;      ';      (;      );      *;      +;      ,;      -;      .;      /;      0;      1;      2;      3;      4;      5;      6;      7;      8;      9;      :;      ;;      <;      =;      >;      ?;      @;      A;      B;      C;      D;      E;      F;      G;      H;      I;      J;      K;      L;      M;      N;      O;      P;      Q;      R;      S;      T;      U;      V;      W;      X;      Y;      Z;      [;      \;      ];      ^;      _;      `;      a;      b;      c;      d;      e;      f;      g;      h;      i;      j;      k;      l;      m;      n;      o;      p;      q;      r;      s;      t;      u;      v;      w;      x;      y;      z;      {;      |;      };      ~;      ;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;      �;       <      <      <      <      <      <      <      <      <      	<      
<      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <       <      !<      "<      #<      $<      %<      &<      '<      (<      )<      *<      +<      ,<      -<      .<      /<      0<      1<      2<      3<      4<      5<      6<      7<      8<      9<      :<      ;<      <<      =<      ><      ?<      @<      A<      B<      C<      D<      E<      F<      G<      H<      I<      J<      K<      L<      M<      N<      O<      P<      Q<      R<      S<      T<      U<      V<      W<      X<      Y<      Z<      [<      \<      ]<      ^<      _<      `<      a<      b<      c<      d<      e<      f<      g<      h<      i<      j<      k<      l<      m<      n<      o<      p<      q<      r<      s<      t<      u<      v<      w<      x<      y<      z<      {<      |<      }<      ~<      <      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<      �<       =      =      =      =      =      =      =      =      =      	=      
=      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =       =      !=      "=      #=      $=      %=      &=      '=      (=      )=      *=      +=      ,=      -=      .=      /=      0=      1=      2=      3=      4=      5=      6=      7=      8=      9=      :=      ;=      <=      ==      >=      ?=      @=      A=      B=      C=      D=      E=      F=      G=      H=      I=      J=      K=      L=      M=      N=      O=      P=      Q=      R=      S=      T=      U=      V=      W=      X=      Y=      Z=      [=      \=      ]=      ^=      _=      `=      a=      b=      c=      d=      e=      f=      g=      h=      i=      j=      k=      l=      m=      n=      o=      p=      q=      r=      s=      t=      u=      v=      w=      x=      y=      z=      {=      |=      }=      ~=      =      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=      �=       >      >      >      >      >      >      >      >      >      	>      
>      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >       >      !>      ">      #>      $>      %>      &>      '>      (>      )>      *>      +>      ,>      ->      .>      />      0>      1>      2>      3>      4>      5>      6>      7>      8>      9>      :>      ;>      <>      =>      >>      ?>      @>      A>      B>      C>      D>      E>      F>      G>      H>      I>      J>      K>      L>      M>      N>      O>      P>      Q>      R>      S>      T>      U>      V>      W>      X>      Y>      Z>      [>      \>      ]>      ^>      _>      `>      a>      b>      c>      d>      e>      f>      g>      h>      i>      j>      k>      l>      m>      n>      o>      p>      q>      r>      s>      t>      u>      v>      w>      x>      y>      z>      {>      |>      }>      ~>      >      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>      �>       ?      ?      ?      ?      ?      ?      ?      ?      ?      	?      
?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       ?      !?      "?      #?      $?      %?      &?      '?      (?      )?      *?      +?      ,?      -?      .?      /?      0?      1?      2?      3?      4?      5?      6?      7?      8?      9?      :?      ;?      <?      =?      >?      ??      @?      A?      B?      C?      D?      E?      F?      G?      H?      I?      J?      K?      L?      M?      N?      O?      P?      Q?      R?      S?      T?      U?      V?      W?      X?      Y?      Z?      [?      \?      ]?      ^?      _?      `?      a?      b?      c?      d?      e?      f?      g?      h?      i?      j?      k?      l?      m?      n?      o?      p?      q?      r?      s?      t?      u?      v?      w?      x?      y?      z?      {?      |?      }?      ~?      ?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?       @      @      @      @      @      @      @      @      @      	@      
@      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @       @      !@      "@      #@      $@      %@      &@      '@      (@      )@      *@      +@      ,@      -@      .@      /@      0@      1@      2@      3@      4@      5@      6@      7@      8@      9@      :@      ;@      <@      =@      >@      ?@      @@      A@      B@      C@      D@      E@      F@      G@      H@      I@      J@      K@      L@      M@      N@      O@      P@      Q@      R@      S@      T@      U@      V@      W@      X@      Y@      Z@      [@      \@      ]@      ^@      _@      `@      a@      b@      c@      d@      e@      f@      g@      h@      i@      j@      k@      l@      m@      n@      o@      p@      q@      r@      s@      t@      u@      v@      w@      x@      y@      z@      {@      |@      }@      ~@      @      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@      �@       A      A      A      A      A      A      A      A      A      	A      
A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A       A      !A      "A      #A      $A      %A      &A      'A      (A      )A      *A      +A      ,A      -A      .A      /A      0A      1A      2A      3A      4A      5A      6A      7A      8A      9A      :A      ;A      <A      =A      >A      ?A      @A      AA      BA      CA      DA      EA      FA      GA      HA      IA      JA      KA      LA      MA      NA      OA      PA      QA      RA      SA      TA      UA      VA      WA      XA      YA      ZA      [A      \A      ]A      ^A      _A      `A      aA      bA      cA      dA      eA      fA      gA      hA      iA      jA      kA      lA      mA      nA      oA      pA      qA      rA      sA      tA      uA      vA      wA      xA      yA      zA      {A      |A      }A      ~A      A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A      �A       B      B      B      B      B      B      B      B      B      	B      
B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B       B      !B      "B      #B      $B      %B      &B      'B      (B      )B      *B      +B      ,B      -B      .B      /B      0B      1B      2B      3B      4B      5B      6B      7B      8B      9B      :B      ;B      <B      =B      >B      ?B      @B      AB      BB      CB      DB      EB      FB      GB      HB      IB      JB      KB      LB      MB      NB      OB      PB      QB      RB      SB      TB      UB      VB      WB      XB      YB      ZB      [B      \B      ]B      ^B      _B      `B      aB      bB      cB      dB      eB      fB      gB      hB      iB      jB      kB      lB      mB      nB      oB      pB      qB      rB      sB      tB      uB      vB      wB      xB      yB      zB      {B      |B      }B      ~B      B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B      �B       C      C      C      C      C      C      C      C      C      	C      
C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C       C      !C      "C      #C      $C      %C      &C      'C      (C      )C      *C      +C      ,C      -C      .C      /C      0C      1C      2C      3C      4C      5C      6C      7C      8C      9C      :C      ;C      <C      =C      >C      ?C      @C      AC      BC      CC      DC      EC      FC      GC      HC      IC      JC      KC      LC      MC      NC      OC      PC      QC      RC      SC      TC      UC      VC      WC      XC      YC      ZC      [C      \C      ]C      ^C      _C      `C      aC      bC      cC      dC      eC      fC      gC      hC      iC      jC      kC      lC      mC      nC      oC      pC      qC      rC      sC      tC      uC      vC      wC      xC      yC      zC      {C      |C      }C      ~C      C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C      �C       D      D      D      D      D      D      D      D      D      	D      
D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D       D      !D      "D      #D      $D      %D      &D      'D      (D      )D      *D      +D      ,D      -D      .D      /D      0D      1D      2D      3D      4D      5D      6D      7D      8D      9D      :D      ;D      <D      =D      >D      ?D      @D      AD      BD      CD      DD      ED      FD      GD      HD      ID      JD      KD      LD      MD      ND      OD      PD      QD      RD      SD      TD      UD      VD      WD      XD      YD      ZD      [D      \D      ]D      ^D      _D      `D      aD      bD      cD      dD      eD      fD      gD      hD      iD      jD      kD      lD      mD      nD      oD      pD      qD      rD      sD      tD      uD      vD      wD      xD      yD      zD      {D      |D      }D      ~D      D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D      �D       E      E      E      E      E      E      E      E      E      	E      
E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E       E      !E      "E      #E      $E      %E      &E      'E      (E      )E      *E      +E      ,E      -E      .E      /E      0E      1E      2E      3E      4E      5E      6E      7E      8E      9E      :E      ;E      <E      =E      >E      ?E      @E      AE      BE      CE      DE      EE      FE      GE      HE      IE      JE      KE      LE      ME      NE      OE      PE      QE      RE      SE      TE      UE      VE      WE      XE      YE      ZE      [E      \E      ]E      ^E      _E      `E      aE      bE      cE      dE      eE      fE      gE      hE      iE      jE      kE      lE      mE      nE      oE      pE      qE      rE      sE      tE      uE      vE      wE      xE      yE      zE      {E      |E      }E      ~E      E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E      �E       F      F      F      F      F      F      F      F      F      	F      
F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F       F      !F      "F      #F      $F      %F      &F      'F      (F      )F      *F      +F      ,F      -F      .F      /F      0F      1F      2F      3F      4F      5F      6F      7F      8F      9F      :F      ;F      <F      =F      >F      ?F      @F      AF      BF      CF      DF      EF      FF      GF      HF      IF      JF      KF      LF      MF      NF      OF      PF      QF      RF      SF      TF      UF      VF      WF      XF      YF      ZF      [F      \F      ]F      ^F      _F      `F      aF      bF      cF      dF      eF      fF      gF      hF      iF      jF      kF      lF      mF      nF      oF      pF      qF      rF      sF      tF      uF      vF      wF      xF      yF      zF      {F      |F      }F      ~F      F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F      �F       G      G      G      G      G      G      G      G      G      	G      
G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G       G      !G      "G      #G      $G      %G      &G      'G      (G      )G      *G      +G      ,G      -G      .G      /G      0G      1G      2G      3G      4G      5G      6G      7G      8G      9G      :G      ;G      <G      =G      >G      ?G      @G      AG      BG      CG      DG      EG      FG      GG      HG      IG      JG      KG      LG      MG      NG      OG      PG      QG      RG      SG      TG      UG      VG      WG      XG      YG      ZG      [G      \G      ]G      ^G      _G      `G      aG      bG      cG      dG      eG      fG      gG      hG      iG      jG      kG      lG      mG      nG      oG      pG      qG      rG      sG      tG      uG      vG      wG      xG      yG      zG      {G      |G      }G      ~G      G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G      �G       H      H      H      H      H      H      H      H      H      	H      
H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H       H      !H      "H      #H      $H      %H      &H      'H      (H      )H      *H      +H      ,H      -H      .H      /H      0H      1H      2H      3H      4H      5H      6H      7H      8H      9H      :H      ;H      <H      =H      >H      ?H      @H      AH      BH      CH      DH      EH      FH      GH      HH      IH      JH      KH      LH      MH      NH      OH      PH      QH      RH      SH      TH      UH      VH      WH      XH      YH      ZH      [H      \H      ]H      ^H      _H      `H      aH      bH      cH      dH      eH      fH      gH      hH      iH      jH      kH      lH      mH      nH      oH      pH      qH      rH      sH      tH      uH      vH      wH      xH      yH      zH      {H      |H      }H      ~H      H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H      �H       I      I      I      I      I      I      I      I      I      	I      
I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I       I      !I      "I      #I      $I      %I      &I      'I      (I      )I      *I      +I      ,I      -I      .I      /I      0I      1I      2I      3I      4I      5I      6I      7I      8I      9I      :I      ;I      <I      =I      >I      ?I      @I      AI      BI      CI      DI      EI      FI      GI      HI      II      JI      KI      LI      MI      NI      OI      PI      QI      RI      SI      TI      UI      VI      WI      XI      YI      ZI      [I      \I      ]I      ^I      _I      `I      aI      bI      cI      dI      eI      fI      gI      hI      iI      jI      kI      lI      mI      nI      oI      pI      qI      rI      sI      tI      uI      vI      wI      xI      yI      zI      {I      |I      }I      ~I      I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I      �I       J      J      J      J      J      J      J      J      J      	J      
J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J       J      !J      "J      #J      $J      %J      &J      'J      (J      )J      *J      +J      ,J      -J      .J      /J      0J      1J      2J      3J      4J      5J      6J      7J      8J      9J      :J      ;J      <J      =J      >J      ?J      @J      AJ      BJ      CJ      DJ      EJ      FJ      GJ      HJ      IJ      JJ      KJ      LJ      MJ      NJ      OJ      PJ      QJ      RJ      SJ      TJ      UJ      VJ      WJ      XJ      YJ      ZJ      [J      \J      ]J      ^J      _J      `J      aJ      bJ      cJ      dJ      eJ      fJ      gJ      hJ      iJ      jJ      kJ      lJ      mJ      nJ      oJ      pJ      qJ      rJ      sJ      tJ      uJ      vJ      wJ      xJ      yJ      zJ      {J      |J      }J      ~J      J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J      �J       K      K      K      K      K      K      K      K      K      	K      
K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K       K      !K      "K      #K      $K      %K      &K      'K      (K      )K      *K      +K      ,K      -K      .K      /K      0K      1K      2K      3K      4K      5K      6K      7K      8K      9K      :K      ;K      <K      =K      >K      ?K      @K      AK      BK      CK      DK      EK      FK      GK      HK      IK      JK      KK      LK      MK      NK      OK      PK      QK      RK      SK      TK      UK      VK      WK      XK      YK      ZK      [K      \K      ]K      ^K      _K      `K      aK      bK      cK      dK      eK      fK      gK      hK      iK      jK      kK      lK      mK      nK      oK      pK      qK      rK      sK      tK      uK      vK      wK      xK      yK      zK      {K      |K      }K      ~K      K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K      �K       L      L      L      L      L      L      L      L      L      	L      
L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L       L      !L      "L      #L      $L      %L      &L      'L      (L      )L      *L      +L      ,L      -L      .L      /L      0L      1L      2L      3L      4L      5L      6L      7L      8L      9L      :L      ;L      <L      =L      >L      ?L      @L      AL      BL      CL      DL      EL      FL      GL      HL      IL      JL      KL      LL      ML      NL      OL      PL      QL      RL      SL      TL      UL      VL      WL      XL      YL      ZL      [L      \L      ]L      ^L      _L      `L      aL      bL      cL      dL      eL      fL      gL      hL      iL      jL      kL      lL      mL      nL      oL      pL      qL      rL      sL      tL      uL      vL      wL      xL      yL      zL      {L      |L      }L      ~L      L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L      �L       M      M      M      M      M      M      M      M      M      	M      
M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M       M      !M      "M      #M      $M      %M      &M      'M      (M      )M      *M      +M      ,M      -M      .M      /M      0M      1M      2M      3M      4M      5M      6M      7M      8M      9M      :M      ;M      <M      =M      >M      ?M      @M      AM      BM      CM      DM      EM      FM      GM      HM      IM      JM      KM      LM      MM      NM      OM      PM      QM      RM      SM      TM      UM      VM      WM      XM      YM      ZM      [M      \M      ]M      ^M      _M      `M      aM      bM      cM      dM      eM      fM      gM      hM      iM      jM      kM      lM      mM      nM      oM      pM      qM      rM      sM      tM      uM      vM      wM      xM      yM      zM      {M      |M      }M      ~M      M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M      �M       N      N      N      N      N      N      N      N      N      	N      
N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N       N      !N      "N      #N      $N      %N      &N      'N      (N      )N      *N      +N      ,N      -N      .N      /N      0N      1N      2N      3N      4N      5N      6N      7N      8N      9N      :N      ;N      <N      =N      >N      ?N      @N      AN      BN      CN      DN      EN      FN      GN      HN      IN      JN      KN      LN      MN      NN      ON      PN      QN      RN      SN      TN      UN      VN      WN      XN      YN      ZN      [N      \N      ]N      ^N      _N      `N      aN      bN      cN      dN      eN      fN      gN      hN      iN      jN      kN      lN      mN      nN      oN      pN      qN      rN      sN      tN      uN      vN      wN      xN      yN      zN      {N      |N      }N      ~N      N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N      �N       O      O      O      O      O      O      O      O      O      	O      
O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O       O      !O      "O      #O      $O      %O      &O      'O      (O      )O      *O      +O      ,O      -O      .O      /O      0O      1O      2O      3O      4O      5O      6O      7O      8O      9O      :O      ;O      <O      =O      >O      ?O      @O      AO      BO      CO      DO      EO      FO      GO      HO      IO      JO      KO      LO      MO      NO      OO      PO      QO      RO      SO      TO      UO      VO      WO      XO      YO      ZO      [O      \O      ]O      ^O      _O      `O      aO      bO      cO      dO      eO      fO      gO      hO      iO      jO      kO      lO      mO      nO      oO      pO      qO      rO      sO      tO      uO      vO      wO      xO      yO      zO      {O      |O      }O      ~O      O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O      �O       P      P      P      P      P      P      P      P      P      	P      
P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P       P      !P      "P      #P      $P      %P      &P      'P      (P      )P      *P      +P      ,P      -P      .P      /P      0P      1P      2P      3P      4P      5P      6P      7P      8P      9P      :P      ;P      <P      =P      >P      ?P      @P      AP      BP      CP      DP      EP      FP      GP      HP      IP      JP      KP      LP      MP      NP      OP      PP      QP      RP      SP      TP      UP      VP      WP      XP      YP      ZP      [P      \P      ]P      ^P      _P      `P      aP      bP      cP      dP      eP      fP      gP      hP      iP      jP      kP      lP      mP      nP      oP      pP      qP      rP      sP      tP      uP      vP      wP      xP      yP      zP      {P      |P      }P      ~P      P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P      �P       Q      Q      Q      Q      Q      Q      Q      Q      Q      	Q      
Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q       Q      !Q      "Q      #Q      $Q      %Q      &Q      'Q      (Q      )Q      *Q      +Q      ,Q      -Q      .Q      /Q      0Q      1Q      2Q      3Q      4Q      5Q      6Q      7Q      8Q      9Q      :Q      ;Q      <Q      =Q      >Q      ?Q      @Q      AQ      BQ      CQ      DQ      EQ      FQ      GQ      HQ      IQ      JQ      KQ      LQ      MQ      NQ      OQ      PQ      QQ      RQ      SQ      TQ      UQ      VQ      WQ      XQ      YQ      ZQ      [Q      \Q      ]Q      ^Q      _Q      `Q      aQ      bQ      cQ      dQ      eQ      fQ      gQ      hQ      iQ      jQ      kQ      lQ      mQ      nQ      oQ      pQ      qQ      rQ      sQ      tQ      uQ      vQ      wQ      xQ      yQ      zQ      {Q      |Q      }Q      ~Q      Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q      �Q       R      R      R      R      R      R      R      R      R      	R      
R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R       R      !R      "R      #R      $R      %R      &R      'R      (R      )R      *R      +R      ,R      -R      .R      /R      0R      1R      2R      3R      4R      5R      6R      7R      8R      9R      :R      ;R      <R      =R      >R      ?R      @R      AR      BR      CR      DR      ER      FR      GR      HR      IR      JR      KR      LR      MR      NR      OR      PR      QR      RR      SR      TR      UR      VR      WR      XR      YR      ZR      [R      \R      ]R      ^R      _R      `R      aR      bR      cR      dR      eR      fR      gR      hR      iR      jR      kR      lR      mR      nR      oR      pR      qR      rR      sR      tR      uR      vR      wR      xR      yR      zR      {R      |R      }R      ~R      R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R      �R       S      S      S      S      S      S      S      S      S      	S      
S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S       S      !S      "S      #S      $S      %S      &S      'S      (S      )S      *S      +S      ,S      -S      .S      /S      0S      1S      2S      3S      4S      5S      6S      7S      8S      9S      :S      ;S      <S      =S      >S      ?S      @S      AS      BS      CS      DS      ES      FS      GS      HS      IS      JS      KS      LS      MS      NS      OS      PS      QS      RS      SS      TS      US      VS      WS      XS      YS      ZS      [S      \S      ]S      ^S      _S      `S      aS      bS      cS      dS      eS      fS      gS      hS      iS      jS      kS      lS      mS      nS      oS      pS      qS      rS      sS      tS      uS      vS      wS      xS      yS      zS      {S      |S      }S      ~S      S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S      �S       T      T      T      T      T      T      T      T      T      	T      
T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T       T      !T      "T      #T      $T      %T      &T      'T      (T      )T      *T      +T      ,T      -T      .T      /T      0T      1T      2T      3T      4T      5T      6T      7T      8T      9T      :T      ;T      <T      =T      >T      ?T      @T      AT      BT      CT      DT      ET      FT      GT      HT      IT      JT      KT      LT      MT      NT      OT      PT      QT      RT      ST      TT      UT      VT      WT      XT      YT      ZT      [T      \T      ]T      ^T      _T      `T      aT      bT      cT      dT      eT      fT      gT      hT      iT      jT      kT      lT      mT      nT      oT      pT      qT      rT      sT      tT      uT      vT      wT      xT      yT      zT      {T      |T      }T      ~T      T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T      �T       U      U      U      U      U      U      U      U      U      	U      
U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U       U      !U      "U      #U      $U      %U      &U      'U      (U      )U      *U      +U      ,U      -U      .U      /U      0U      1U      2U      3U      4U      5U      6U      7U      8U      9U      :U      ;U      <U      =U      >U      ?U      @U      AU      BU      CU      DU      EU      FU      GU      HU      IU      JU      KU      LU      MU      NU      OU      PU      QU      RU      SU      TU      UU      VU      WU      XU      YU      ZU      [U      \U      ]U      ^U      _U      `U      aU      bU      cU      dU      eU      fU      gU      hU      iU      jU      kU      lU      mU      nU      oU      pU      qU      rU      sU      tU      uU      vU      wU      xU      yU      zU      {U      |U      }U      ~U      U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U      �U       V      V      V      V      V      V      V      V      V      	V      
V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V       V      !V      "V      #V      $V      %V      &V      'V      (V      )V      *V      +V      ,V      -V      .V      /V      0V      1V      2V      3V      4V      5V      6V      7V      8V      9V      :V      ;V      <V      =V      >V      ?V      @V      AV      BV      CV      DV      EV      FV      GV      HV      IV      JV      KV      LV      MV      NV      OV      PV      QV      RV      SV      TV      UV      VV      WV      XV      YV      ZV      [V      \V      ]V      ^V      _V      `V      aV      bV      cV      dV      eV      fV      gV      hV      iV      jV      kV      lV      mV      nV      oV      pV      qV      rV      sV      tV      uV      vV      wV      xV      yV      zV      {V      |V      }V      ~V      V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V      �V       W      W      W      W      W      W      W      W      W      	W      
W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W       W      !W      "W      #W      $W      %W      &W      'W      (W      )W      *W      +W      ,W      -W      .W      /W      0W      1W      2W      3W      4W      5W      6W      7W      8W      9W      :W      ;W      <W      =W      >W      ?W      @W      AW      BW      CW      DW      EW      FW      GW      HW      IW      JW      KW      LW      MW      NW      OW      PW      QW      RW      SW      TW      UW      VW      WW      XW      YW      ZW      [W      \W      ]W      ^W      _W      `W      aW      bW      cW      dW      eW      fW      gW      hW      iW      jW      kW      lW      mW      nW      oW      pW      qW      rW      sW      tW      uW      vW      wW      xW      yW      zW      {W      |W      }W      ~W      W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W      �W       X      X      X      X      X      X      X      X      X      	X      
X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X       X      !X      "X      #X      $X      %X      &X      'X      (X      )X      *X      +X      ,X      -X      .X      /X      0X      1X      2X      3X      4X      5X      6X      7X      8X      9X      :X      ;X      <X      =X      >X      ?X      @X      AX      BX      CX      DX      EX      FX      GX      HX      IX      JX      KX      LX      MX      NX      OX      PX      QX      RX      SX      TX      UX      VX      WX      XX      YX      ZX      [X      \X      ]X      ^X      _X      `X      aX      bX      cX      dX      eX      fX      gX      hX      iX      jX      kX      lX      mX      nX      oX      pX      qX      rX      sX      tX      uX      vX      wX      xX      yX      zX      {X      |X      }X      ~X      X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X      �X       Y      Y      Y      Y      Y      Y      Y      Y      Y      	Y      
Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y       Y      !Y      "Y      #Y      $Y      %Y      &Y      'Y      (Y      )Y      *Y      +Y      ,Y      -Y      .Y      /Y      0Y      1Y      2Y      3Y      4Y      5Y      6Y      7Y      8Y      9Y      :Y      ;Y      <Y      =Y      >Y      ?Y      @Y      AY      BY      CY      DY      EY      FY      GY      HY      IY      JY      KY      LY      MY      NY      OY      PY      QY      RY      SY      TY      UY      VY      WY      XY      YY      ZY      [Y      \Y      ]Y      ^Y      _Y      `Y      aY      bY      cY      dY      eY      fY      gY      hY      iY      jY      kY      lY      mY      nY      oY      pY      qY      rY      sY      tY      uY      vY      wY      xY      yY      zY      {Y      |Y      }Y      ~Y      Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y      �Y       Z      Z      Z      Z      Z      Z      Z      Z      Z      	Z      
Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z       Z      !Z      "Z      #Z      $Z      %Z      &Z      'Z      (Z      )Z      *Z      +Z      ,Z      -Z      .Z      /Z      0Z      1Z      2Z      3Z      4Z      5Z      6Z      7Z      8Z      9Z      :Z      ;Z      <Z      =Z      >Z      ?Z      @Z      AZ      BZ      CZ      DZ      EZ      FZ      GZ      HZ      IZ      JZ      KZ      LZ      MZ      NZ      OZ      PZ      QZ      RZ      SZ      TZ      UZ      VZ      WZ      XZ      YZ      ZZ      [Z      \Z      ]Z      ^Z      _Z      `Z      aZ      bZ      cZ      dZ      eZ      fZ      gZ      hZ      iZ      jZ      kZ      lZ      mZ      nZ      oZ      pZ      qZ      rZ      sZ      tZ      uZ      vZ      wZ      xZ      yZ      zZ      {Z      |Z      }Z      ~Z      Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z      �Z       [      [      [      [      [      [      [      [      [      	[      
[      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [       [      ![      "[      #[      $[      %[      &[      '[      ([      )[      *[      +[      ,[      -[      .[      /[      0[      1[      2[      3[      4[      5[      6[      7[      8[      9[      :[      ;[      <[      =[      >[      ?[      @[      A[      B[      C[      D[      E[      F[      G[      H[      I[      J[      K[      L[      M[      N[      O[      P[      Q[      R[      S[      T[      U[      V[      W[      X[      Y[      Z[      [[      \[      ][      ^[      _[      `[      a[      b[      c[      d[      e[      f[      g[      h[      i[      j[      k[      l[      m[      n[      o[      p[      q[      r[      s[      t[      u[      v[      w[      x[      y[      z[      {[      |[      }[      ~[      [      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[      �[       \      \      \      \      \      \      \      \      \      	\      
\      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \       \      !\      "\      #\      $\      %\      &\      '\      (\      )\      *\      +\      ,\      -\      .\      /\      0\      1\      2\      3\      4\      5\      6\      7\      8\      9\      :\      ;\      <\      =\      >\      ?\      @\      A\      B\      C\      D\      E\      F\      G\      H\      I\      J\      K\      L\      M\      N\      O\      P\      Q\      R\      S\      T\      U\      V\      W\      X\      Y\      Z\      [\      \\      ]\      ^\      _\      `\      a\      b\      c\      d\      e\      f\      g\      h\      i\      j\      k\      l\      m\      n\      o\      p\      q\      r\      s\      t\      u\      v\      w\      x\      y\      z\      {\      |\      }\      ~\      \      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\      �\       ]      ]      ]      ]      ]      ]      ]      ]      ]      	]      
]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]       ]      !]      "]      #]      $]      %]      &]      ']      (]      )]      *]      +]      ,]      -]      .]      /]      0]      1]      2]      3]      4]      5]      6]      7]      8]      9]      :]      ;]      <]      =]      >]      ?]      @]      A]      B]      C]      D]      E]      F]      G]      H]      I]      J]      K]      L]      M]      N]      O]      P]      Q]      R]      S]      T]      U]      V]      W]      X]      Y]      Z]      []      \]      ]]      ^]      _]      `]      a]      b]      c]      d]      e]      f]      g]      h]      i]      j]      k]      l]      m]      n]      o]      p]      q]      r]      s]      t]      u]      v]      w]      x]      y]      z]      {]      |]      }]      ~]      ]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]      �]       ^      ^      ^      ^      ^      ^      ^      ^      ^      	^      
^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^       ^      !^      "^      #^      $^      %^      &^      '^      (^      )^      *^      +^      ,^      -^      .^      /^      0^      1^      2^      3^      4^      5^      6^      7^      8^      9^      :^      ;^      <^      =^      >^      ?^      @^      A^      B^      C^      D^      E^      F^      G^      H^      I^      J^      K^      L^      M^      N^      O^      P^      Q^      R^      S^      T^      U^      V^      W^      X^      Y^      Z^      [^      \^      ]^      ^^      _^      `^      a^      b^      c^      d^      e^      f^      g^      h^      i^      j^      k^      l^      m^      n^      o^      p^      q^      r^      s^      t^      u^      v^      w^      x^      y^      z^      {^      |^      }^      ~^      ^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^      �^       _      _      _      _      _      _      _      _      _      	_      
_      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _       _      !_      "_      #_      $_      %_      &_      '_      (_      )_      *_      +_      ,_      -_      ._      /_      0_      1_      2_      3_      4_      5_      6_      7_      8_      9_      :_      ;_      <_      =_      >_      ?_      @_      A_      B_      C_      D_      E_      F_      G_      H_      I_      J_      K_      L_      M_      N_      O_      P_      Q_      R_      S_      T_      U_      V_      W_      X_      Y_      Z_      [_      \_      ]_      ^_      __      `_      a_      b_      c_      d_      e_      f_      g_      h_      i_      j_      k_      l_      m_      n_      o_      p_      q_      r_      s_      t_      u_      v_      w_      x_      y_      z_      {_      |_      }_      ~_      _      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_      �_       `      `      `      `      `      `      `      `      `      	`      
`      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `       `      !`      "`      #`      $`      %`      &`      '`      (`      )`      *`      +`      ,`      -`      .`      /`      0`      1`      2`      3`      4`      5`      6`      7`      8`      9`      :`      ;`      <`      =`      >`      ?`      @`      A`      B`      C`      D`      E`      F`      G`      H`      I`      J`      K`      L`      M`      N`      O`      P`      Q`      R`      S`      T`      U`      V`      W`      X`      Y`      Z`      [`      \`      ]`      ^`      _`      ``      a`      b`      c`      d`      e`      f`      g`      h`      i`      j`      k`      l`      m`      n`      o`      p`      q`      r`      s`      t`      u`      v`      w`      x`      y`      z`      {`      |`      }`      ~`      `      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`      �`       a      a      a      a      a      a      a      a      a      	a      
a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a       a      !a      "a      #a      $a      %a      &a      'a      (a      )a      *a      +a      ,a      -a      .a      /a      0a      1a      2a      3a      4a      5a      6a      7a      8a      9a      :a      ;a      <a      =a      >a      ?a      @a      Aa      Ba      Ca      Da      Ea      Fa      Ga      Ha      Ia      Ja      Ka      La      Ma      Na      Oa      Pa      Qa      Ra      Sa      Ta      Ua      Va      Wa      Xa      Ya      Za      [a      \a      ]a      ^a      _a      `a      aa      ba      ca      da      ea      fa      ga      ha      ia      ja      ka      la      ma      na      oa      pa      qa      ra      sa      ta      ua      va      wa      xa      ya      za      {a      |a      }a      ~a      a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a      �a       b      b      b      b      b      b      b      b      b      	b      
b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b       b      !b      "b      #b      $b      %b      &b      'b      (b      )b      *b      +b      ,b      -b      .b      /b      0b      1b      2b      3b      4b      5b      6b      7b      8b      9b      :b      ;b      <b      =b      >b      ?b      @b      Ab      Bb      Cb      Db      Eb      Fb      Gb      Hb      Ib      Jb      Kb      Lb      Mb      Nb      Ob      Pb      Qb      Rb      Sb      Tb      Ub      Vb      Wb      Xb      Yb      Zb      [b      \b      ]b      ^b      _b      `b      ab      bb      cb      db      eb      fb      gb      hb      ib      jb      kb      lb      mb      nb      ob      pb      qb      rb      sb      tb      ub      vb      wb      xb      yb      zb      {b      |b      }b      ~b      b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b      �b       c      c      c      c      c      c      c      c      c      	c      
c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c       c      !c      "c      #c      $c      %c      &c      'c      (c      )c      *c      +c      ,c      -c      .c      /c      0c      1c      2c      3c      4c      5c      6c      7c      8c      9c      :c      ;c      <c      =c      >c      ?c      @c      Ac      Bc      Cc      Dc      Ec      Fc      Gc      Hc      Ic      Jc      Kc      Lc      Mc      Nc      Oc      Pc      Qc      Rc      Sc      Tc      Uc      Vc      Wc      Xc      Yc      Zc      [c      \c      ]c      ^c      _c      `c      ac      bc      cc      dc      ec      fc      gc      hc      ic      jc      kc      lc      mc      nc      oc      pc      qc      rc      sc      tc      uc      vc      wc      xc      yc      zc      {c      |c      }c      ~c      c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c      �c       d      d      d      d      d      d      d      d      d      	d      
d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d       d      !d      "d      #d      $d      %d      &d      'd      (d      )d      *d      +d      ,d      -d      .d      /d      0d      1d      2d      3d      4d      5d      6d      7d      8d      9d      :d      ;d      <d      =d      >d      ?d      @d      Ad      Bd      Cd      Dd      Ed      Fd      Gd      Hd      Id      Jd      Kd      Ld      Md      Nd      Od      Pd      Qd      Rd      Sd      Td      Ud      Vd      Wd      Xd      Yd      Zd      [d      \d      ]d      ^d      _d      `d      ad      bd      cd      dd      ed      fd      gd      hd      id      jd      kd      ld      md      nd      od      pd      qd      rd      sd      td      ud      vd      wd      xd      yd      zd      {d      |d      }d      ~d      d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d      �d       e      e      e      e      e      e      e      e      e      	e      
e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e       e      !e      "e      #e      $e      %e      &e      'e      (e      )e      *e      +e      ,e      -e      .e      /e      0e      1e      2e      3e      4e      5e      6e      7e      8e      9e      :e      ;e      <e      =e      >e      ?e      @e      Ae      Be      Ce      De      Ee      Fe      Ge      He      Ie      Je      Ke      Le      Me      Ne      Oe      Pe      Qe      Re      Se      Te      Ue      Ve      We      Xe      Ye      Ze      [e      \e      ]e      ^e      _e      `e      ae      be      ce      de      ee      fe      ge      he      ie      je      ke      le      me      ne      oe      pe      qe      re      se      te      ue      ve      we      xe      ye      ze      {e      |e      }e      ~e      e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e      �e       f      f      f      f      f      f      f      f      f      	f      
f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f       f      !f      "f      #f      $f      %f      &f      'f      (f      )f      *f      +f      ,f      -f      .f      /f      0f      1f      2f      3f      4f      5f      6f      7f      8f      9f      :f      ;f      <f      =f      >f      ?f      @f      Af      Bf      Cf      Df      Ef      Ff      Gf      Hf      If      Jf      Kf      Lf      Mf      Nf      Of      Pf      Qf      Rf      Sf      Tf      Uf      Vf      Wf      Xf      Yf      Zf      [f      \f      ]f      ^f      _f      `f      af      bf      cf      df      ef      ff      gf      hf      if      jf      kf      lf      mf      nf      of      pf      qf      rf      sf      tf      uf      vf      wf      xf      yf      zf      {f      |f      }f      ~f      f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      �f      
�
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *$
fR
__inference_<lambda>_324818
�
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
 *0
config_proto 

CPU

GPU2*0J 8� *$
fR
__inference_<lambda>_324823
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
�5
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*�4
value�4B�4 B�4
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
;
	keras_api
_lookup_layer
_adapt_function*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
'
1
%2
&3
-4
.5*
* 
* 
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
4trace_0
5trace_1
6trace_2
7trace_3* 
6
8trace_0
9trace_1
:trace_2
;trace_3* 
* 
�
<iter

=beta_1

>beta_2
	?decay
@learning_rate
Amomentum_cachem{%m|&m}-m~.mv�%v�&v�-v�.v�*

Bserving_default* 
* 
7
C	keras_api
Dlookup_table
Etoken_counts*

Ftrace_0* 

0*
* 
* 
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ltrace_0* 

Mtrace_0* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Strace_0* 

Ttrace_0* 

%0
&1*
* 
* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*
* 
* 
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

atrace_0* 

btrace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
'
1
%2
&3
-4
.5*
'
0
1
2
3
4*

c0
d1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
MG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
R
e_initializer
f_create_resource
g_initialize
h_destroy_resource* 
�
i_create_resource
j_initialize
k_destroy_resource><layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/*
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

%0
&1*
* 
* 
* 
* 
* 
* 

-0
.1*
* 
* 
* 
* 
* 
* 
8
l	variables
m	keras_api
	ntotal
	ocount*
H
p	variables
q	keras_api
	rtotal
	scount
t
_fn_kwargs*
* 

utrace_0* 

vtrace_0* 

wtrace_0* 

xtrace_0* 

ytrace_0* 

ztrace_0* 

n0
o1*

l	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

r0
s1*

p	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
��
VARIABLE_VALUENadam/embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUENadam/embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUENadam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
(serving_default_text_vectorization_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCall_1StatefulPartitionedCall(serving_default_text_vectorization_input
hash_tableConstConst_1Const_2embedding/embeddingsdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_324491
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0Nadam/embedding/embeddings/m/Read/ReadVariableOp*Nadam/dense_2/kernel/m/Read/ReadVariableOp(Nadam/dense_2/bias/m/Read/ReadVariableOp*Nadam/dense_3/kernel/m/Read/ReadVariableOp(Nadam/dense_3/bias/m/Read/ReadVariableOp0Nadam/embedding/embeddings/v/Read/ReadVariableOp*Nadam/dense_2/kernel/v/Read/ReadVariableOp(Nadam/dense_2/bias/v/Read/ReadVariableOp*Nadam/dense_3/kernel/v/Read/ReadVariableOp(Nadam/dense_3/bias/v/Read/ReadVariableOpConst_6*(
Tin!
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_324935
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding/embeddingsdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cacheMutableHashTabletotal_1count_1totalcountNadam/embedding/embeddings/mNadam/dense_2/kernel/mNadam/dense_2/bias/mNadam/dense_3/kernel/mNadam/dense_3/bias/mNadam/embedding/embeddings/vNadam/dense_2/kernel/vNadam/dense_2/bias/vNadam/dense_3/kernel/vNadam/dense_3/bias/v*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_325023��	
�p
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324683

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	6
!embedding_embedding_lookup_324661:���9
&dense_2_matmul_readvariableop_resource:	� 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�embedding/embedding_lookup�Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite �
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������W      �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:����������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_324661?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*4
_class*
(&loc:@embedding/embedding_lookup/324661*-
_output_shapes
:�����������*
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/324661*-
_output_shapes
:������������
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:�����������s
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_average_pooling1d_1/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
dense_2/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding/embedding_lookupE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2�
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
$__inference_signature_wrapper_324491
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:���
	unknown_4:	� 
	unknown_5: 
	unknown_6: 
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_324027o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
'
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_324037

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�i
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324284

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	%
embedding_324269:���!
dense_2_324273:	� 
dense_2_324275:  
dense_3_324278: 
dense_3_324280:
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite �
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������W      �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:����������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_324269*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_324104�
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_324037�
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_324273dense_2_324275*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_324120�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_324278dense_3_324280*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_324137w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2�
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�C
�
__inference_adapt_step_323250
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	��IteratorGetNext�(None_lookup_table_find/LookupTableFindV2�,None_lookup_table_insert/LookupTableInsertV2�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:���������*&
output_shapes
:���������*
output_types
2a
StringLowerStringLowerIteratorGetNext:components:0*'
_output_shapes
:����������
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite }
SqueezeSqueezeStaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
StringSplit/StringSplitV2StringSplitV2Squeeze:output:0StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:���������:���������:���������*
out_idx0	�
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:�
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
�

�
C__inference_dense_3_layer_call_and_return_conditional_losses_324750

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_324027
text_vectorization_inputb
^sequential_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handlec
_sequential_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	9
5sequential_1_text_vectorization_string_lookup_equal_y<
8sequential_1_text_vectorization_string_lookup_selectv2_t	C
.sequential_1_embedding_embedding_lookup_324005:���F
3sequential_1_dense_2_matmul_readvariableop_resource:	� B
4sequential_1_dense_2_biasadd_readvariableop_resource: E
3sequential_1_dense_3_matmul_readvariableop_resource: B
4sequential_1_dense_3_biasadd_readvariableop_resource:
identity��+sequential_1/dense_2/BiasAdd/ReadVariableOp�*sequential_1/dense_2/MatMul/ReadVariableOp�+sequential_1/dense_3/BiasAdd/ReadVariableOp�*sequential_1/dense_3/MatMul/ReadVariableOp�'sequential_1/embedding/embedding_lookup�Qsequential_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2}
+sequential_1/text_vectorization/StringLowerStringLowertext_vectorization_input*'
_output_shapes
:����������
2sequential_1/text_vectorization/StaticRegexReplaceStaticRegexReplace4sequential_1/text_vectorization/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite �
'sequential_1/text_vectorization/SqueezeSqueeze;sequential_1/text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������r
1sequential_1/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
9sequential_1/text_vectorization/StringSplit/StringSplitV2StringSplitV20sequential_1/text_vectorization/Squeeze:output:0:sequential_1/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
?sequential_1/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
Asequential_1/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
Asequential_1/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
9sequential_1/text_vectorization/StringSplit/strided_sliceStridedSliceCsequential_1/text_vectorization/StringSplit/StringSplitV2:indices:0Hsequential_1/text_vectorization/StringSplit/strided_slice/stack:output:0Jsequential_1/text_vectorization/StringSplit/strided_slice/stack_1:output:0Jsequential_1/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
Asequential_1/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Csequential_1/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Csequential_1/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential_1/text_vectorization/StringSplit/strided_slice_1StridedSliceAsequential_1/text_vectorization/StringSplit/StringSplitV2:shape:0Jsequential_1/text_vectorization/StringSplit/strided_slice_1/stack:output:0Lsequential_1/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Lsequential_1/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
bsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastBsequential_1/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastDsequential_1/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapefsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
ksequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdusequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0usequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
psequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatertsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ysequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
ksequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastrsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxfsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0wsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ssequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0usequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulosequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumhsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumhsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
osequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountfsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0wsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
isequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumvsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
msequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
isequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2vsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Qsequential_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2^sequential_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleBsequential_1/text_vectorization/StringSplit/StringSplitV2:values:0_sequential_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
3sequential_1/text_vectorization/string_lookup/EqualEqualBsequential_1/text_vectorization/StringSplit/StringSplitV2:values:05sequential_1_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
6sequential_1/text_vectorization/string_lookup/SelectV2SelectV27sequential_1/text_vectorization/string_lookup/Equal:z:08sequential_1_text_vectorization_string_lookup_selectv2_tZsequential_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
6sequential_1/text_vectorization/string_lookup/IdentityIdentity?sequential_1/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������~
<sequential_1/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
4sequential_1/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������W      �
Csequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor=sequential_1/text_vectorization/RaggedToTensor/Const:output:0?sequential_1/text_vectorization/string_lookup/Identity:output:0Esequential_1/text_vectorization/RaggedToTensor/default_value:output:0Dsequential_1/text_vectorization/StringSplit/strided_slice_1:output:0Bsequential_1/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:����������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
'sequential_1/embedding/embedding_lookupResourceGather.sequential_1_embedding_embedding_lookup_324005Lsequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*A
_class7
53loc:@sequential_1/embedding/embedding_lookup/324005*-
_output_shapes
:�����������*
dtype0�
0sequential_1/embedding/embedding_lookup/IdentityIdentity0sequential_1/embedding/embedding_lookup:output:0*
T0*A
_class7
53loc:@sequential_1/embedding/embedding_lookup/324005*-
_output_shapes
:������������
2sequential_1/embedding/embedding_lookup/Identity_1Identity9sequential_1/embedding/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:������������
>sequential_1/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
,sequential_1/global_average_pooling1d_1/MeanMean;sequential_1/embedding/embedding_lookup/Identity_1:output:0Gsequential_1/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:�����������
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
sequential_1/dense_2/MatMulMatMul5sequential_1/global_average_pooling1d_1/Mean:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� z
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
sequential_1/dense_3/MatMulMatMul'sequential_1/dense_2/Relu:activations:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_1/dense_3/SigmoidSigmoid%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������o
IdentityIdentity sequential_1/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp(^sequential_1/embedding/embedding_lookupR^sequential_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2R
'sequential_1/embedding/embedding_lookup'sequential_1/embedding/embedding_lookup2�
Qsequential_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Qsequential_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:a ]
'
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
__inference__initializer_3247637
3key_value_init4489_lookuptableimportv2_table_handle/
+key_value_init4489_lookuptableimportv2_keys1
-key_value_init4489_lookuptableimportv2_values	
identity��&key_value_init4489/LookupTableImportV2�
&key_value_init4489/LookupTableImportV2LookupTableImportV23key_value_init4489_lookuptableimportv2_table_handle+key_value_init4489_lookuptableimportv2_keys-key_value_init4489_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init4489/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :��:��2P
&key_value_init4489/LookupTableImportV2&key_value_init4489/LookupTableImportV2:"

_output_shapes

:��:"

_output_shapes

:��
�
�
*__inference_embedding_layer_call_fn_324690

inputs	
unknown:���
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_324104u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�k
�
"__inference__traced_restore_325023
file_prefix:
%assignvariableop_embedding_embeddings:���4
!assignvariableop_1_dense_2_kernel:	� -
assignvariableop_2_dense_2_bias: 3
!assignvariableop_3_dense_3_kernel: -
assignvariableop_4_dense_3_bias:'
assignvariableop_5_nadam_iter:	 )
assignvariableop_6_nadam_beta_1: )
assignvariableop_7_nadam_beta_2: (
assignvariableop_8_nadam_decay: 0
&assignvariableop_9_nadam_learning_rate: 2
(assignvariableop_10_nadam_momentum_cache: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: #
assignvariableop_13_total: #
assignvariableop_14_count: E
0assignvariableop_15_nadam_embedding_embeddings_m:���=
*assignvariableop_16_nadam_dense_2_kernel_m:	� 6
(assignvariableop_17_nadam_dense_2_bias_m: <
*assignvariableop_18_nadam_dense_3_kernel_m: 6
(assignvariableop_19_nadam_dense_3_bias_m:E
0assignvariableop_20_nadam_embedding_embeddings_v:���=
*assignvariableop_21_nadam_dense_2_kernel_v:	� 6
(assignvariableop_22_nadam_dense_2_bias_v: <
*assignvariableop_23_nadam_dense_3_kernel_v: 6
(assignvariableop_24_nadam_dense_3_bias_v:
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�2MutableHashTable_table_restore/LookupTableImportV2�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_3_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_3_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_nadam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_nadam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_nadam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_nadam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp&assignvariableop_9_nadam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp(assignvariableop_10_nadam_momentum_cacheIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0�
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:11RestoreV2:tensors:12*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_nadam_embedding_embeddings_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_nadam_dense_2_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_nadam_dense_2_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_nadam_dense_3_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_nadam_dense_3_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_nadam_embedding_embeddings_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_nadam_dense_2_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_nadam_dense_2_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_nadam_dense_3_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_nadam_dense_3_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
�
�
__inference_<lambda>_3248187
3key_value_init4489_lookuptableimportv2_table_handle/
+key_value_init4489_lookuptableimportv2_keys1
-key_value_init4489_lookuptableimportv2_values	
identity��&key_value_init4489/LookupTableImportV2�
&key_value_init4489/LookupTableImportV2LookupTableImportV23key_value_init4489_lookuptableimportv2_table_handle+key_value_init4489_lookuptableimportv2_keys-key_value_init4489_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init4489/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :��:��2P
&key_value_init4489/LookupTableImportV2&key_value_init4489/LookupTableImportV2:"

_output_shapes

:��:"

_output_shapes

:��
�
�
E__inference_embedding_layer_call_and_return_conditional_losses_324699

inputs	,
embedding_lookup_324693:���
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_324693inputs*
Tindices0	**
_class 
loc:@embedding_lookup/324693*-
_output_shapes
:�����������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/324693*-
_output_shapes
:������������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:�����������y
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*-
_output_shapes
:�����������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
-
__inference__destroyer_324768
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
E__inference_embedding_layer_call_and_return_conditional_losses_324104

inputs	,
embedding_lookup_324098:���
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_324098inputs*
Tindices0	**
_class 
loc:@embedding_lookup/324098*-
_output_shapes
:�����������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/324098*-
_output_shapes
:������������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:�����������y
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*-
_output_shapes
:�����������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_restore_fn_324810
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity��2MutableHashTable_table_restore/LookupTableImportV2�
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H
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
�<
�
__inference__traced_save_324935
file_prefix3
/savev2_embedding_embeddings_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_nadam_embedding_embeddings_m_read_readvariableop5
1savev2_nadam_dense_2_kernel_m_read_readvariableop3
/savev2_nadam_dense_2_bias_m_read_readvariableop5
1savev2_nadam_dense_3_kernel_m_read_readvariableop3
/savev2_nadam_dense_3_bias_m_read_readvariableop;
7savev2_nadam_embedding_embeddings_v_read_readvariableop5
1savev2_nadam_dense_2_kernel_v_read_readvariableop3
/savev2_nadam_dense_2_bias_v_read_readvariableop5
1savev2_nadam_dense_3_kernel_v_read_readvariableop3
/savev2_nadam_dense_3_bias_v_read_readvariableop
savev2_const_6

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_nadam_embedding_embeddings_m_read_readvariableop1savev2_nadam_dense_2_kernel_m_read_readvariableop/savev2_nadam_dense_2_bias_m_read_readvariableop1savev2_nadam_dense_3_kernel_m_read_readvariableop/savev2_nadam_dense_3_bias_m_read_readvariableop7savev2_nadam_embedding_embeddings_v_read_readvariableop1savev2_nadam_dense_2_kernel_v_read_readvariableop/savev2_nadam_dense_2_bias_v_read_readvariableop1savev2_nadam_dense_3_kernel_v_read_readvariableop/savev2_nadam_dense_3_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 **
dtypes 
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :���:	� : : :: : : : : : ::: : : : :���:	� : : ::���:	� : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:'#
!
_output_shapes
:���:%!

_output_shapes
:	� : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :
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
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :'#
!
_output_shapes
:���:%!

_output_shapes
:	� : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::'#
!
_output_shapes
:���:%!

_output_shapes
:	� : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
�
G
__inference__creator_324773
identity: ��MutableHashTable|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
�
�
__inference_save_fn_324802
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	��?MutableHashTable_lookup_table_export_values/LookupTableExportV2�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::P
add/yConst*
_output_shapes
: *
dtype0*
valueB B
table-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: T
add_1/yConst*
_output_shapes
: *
dtype0*
valueB Btable-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: �

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: �

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:�
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
�

�
-__inference_sequential_1_layer_call_fn_324537

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:���
	unknown_4:	� 
	unknown_5: 
	unknown_6: 
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_324284o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_dense_3_layer_call_fn_324739

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_324137o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
W
;__inference_global_average_pooling1d_1_layer_call_fn_324704

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_324037i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
-
__inference__destroyer_324783
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_324730

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
-__inference_sequential_1_layer_call_fn_324514

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:���
	unknown_4:	� 
	unknown_5: 
	unknown_6: 
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_324144o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�p
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324610

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	6
!embedding_embedding_lookup_324588:���9
&dense_2_matmul_readvariableop_resource:	� 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�embedding/embedding_lookup�Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite �
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������W      �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:����������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_324588?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*4
_class*
(&loc:@embedding/embedding_lookup/324588*-
_output_shapes
:�����������*
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/324588*-
_output_shapes
:������������
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:�����������s
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_average_pooling1d_1/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
dense_2/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding/embedding_lookupE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2�
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_324120

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�i
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324144

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	%
embedding_324105:���!
dense_2_324121:	� 
dense_2_324123:  
dense_3_324138: 
dense_3_324140:
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite �
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������W      �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:����������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_324105*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_324104�
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_324037�
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_324121dense_2_324123*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_324120�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_324138dense_3_324140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_324137w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2�
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�i
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324460
text_vectorization_inputU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	%
embedding_324445:���!
dense_2_324449:	� 
dense_2_324451:  
dense_3_324454: 
dense_3_324456:
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2p
text_vectorization/StringLowerStringLowertext_vectorization_input*'
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite �
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������W      �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:����������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_324445*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_324104�
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_324037�
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_324449dense_2_324451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_324120�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_324454dense_3_324456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_324137w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2�
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:a ]
'
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
-__inference_sequential_1_layer_call_fn_324328
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:���
	unknown_4:	� 
	unknown_5: 
	unknown_6: 
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_324284o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
'
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_324710

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_324719

inputs
unknown:	� 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_324120o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_3_layer_call_and_return_conditional_losses_324137

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
/
__inference__initializer_324778
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
;
__inference__creator_324755
identity��
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name4490*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
�

�
-__inference_sequential_1_layer_call_fn_324165
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:���
	unknown_4:	� 
	unknown_5: 
	unknown_6: 
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_324144o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
'
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�i
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324394
text_vectorization_inputU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	%
embedding_324379:���!
dense_2_324383:	� 
dense_2_324385:  
dense_3_324388: 
dense_3_324390:
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2p
text_vectorization/StringLowerStringLowertext_vectorization_input*'
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite �
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������W      �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:����������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_324379*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_324104�
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_324037�
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_324383dense_2_324385*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_324120�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_324388dense_3_324390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_324137w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2�
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:a ]
'
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
+
__inference_<lambda>_324823
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes "�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
]
text_vectorization_inputA
*serving_default_text_vectorization_input:0���������=
dense_32
StatefulPartitionedCall_1:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
C
1
%2
&3
-4
.5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
4trace_0
5trace_1
6trace_2
7trace_32�
-__inference_sequential_1_layer_call_fn_324165
-__inference_sequential_1_layer_call_fn_324514
-__inference_sequential_1_layer_call_fn_324537
-__inference_sequential_1_layer_call_fn_324328�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z4trace_0z5trace_1z6trace_2z7trace_3
�
8trace_0
9trace_1
:trace_2
;trace_32�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324610
H__inference_sequential_1_layer_call_and_return_conditional_losses_324683
H__inference_sequential_1_layer_call_and_return_conditional_losses_324394
H__inference_sequential_1_layer_call_and_return_conditional_losses_324460�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z8trace_0z9trace_1z:trace_2z;trace_3
�B�
!__inference__wrapped_model_324027text_vectorization_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
<iter

=beta_1

>beta_2
	?decay
@learning_rate
Amomentum_cachem{%m|&m}-m~.mv�%v�&v�-v�.v�"
	optimizer
,
Bserving_default"
signature_map
"
_generic_user_object
L
C	keras_api
Dlookup_table
Etoken_counts"
_tf_keras_layer
�
Ftrace_02�
__inference_adapt_step_323250�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zFtrace_0
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ltrace_02�
*__inference_embedding_layer_call_fn_324690�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0
�
Mtrace_02�
E__inference_embedding_layer_call_and_return_conditional_losses_324699�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zMtrace_0
):'���2embedding/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Strace_02�
;__inference_global_average_pooling1d_1_layer_call_fn_324704�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0
�
Ttrace_02�
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_324710�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_02�
(__inference_dense_2_layer_call_fn_324719�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0
�
[trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_324730�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0
!:	� 2dense_2/kernel
: 2dense_2/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
atrace_02�
(__inference_dense_3_layer_call_fn_324739�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
�
btrace_02�
C__inference_dense_3_layer_call_and_return_conditional_losses_324750�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
 : 2dense_3/kernel
:2dense_3/bias
C
1
%2
&3
-4
.5"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_1_layer_call_fn_324165text_vectorization_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_324514inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_324537inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_324328text_vectorization_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324610inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324683inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324394text_vectorization_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324460text_vectorization_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
�B�
$__inference_signature_wrapper_324491text_vectorization_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
f
e_initializer
f_create_resource
g_initialize
h_destroy_resourceR jtf.StaticHashTable
L
i_create_resource
j_initialize
k_destroy_resourceR Z

 ��
�B�
__inference_adapt_step_323250iterator"�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_embedding_layer_call_fn_324690inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_embedding_layer_call_and_return_conditional_losses_324699inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
;__inference_global_average_pooling1d_1_layer_call_fn_324704inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_324710inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_2_layer_call_fn_324719inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_324730inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_3_layer_call_fn_324739inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_3_layer_call_and_return_conditional_losses_324750inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
l	variables
m	keras_api
	ntotal
	ocount"
_tf_keras_metric
^
p	variables
q	keras_api
	rtotal
	scount
t
_fn_kwargs"
_tf_keras_metric
"
_generic_user_object
�
utrace_02�
__inference__creator_324755�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zutrace_0
�
vtrace_02�
__inference__initializer_324763�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zvtrace_0
�
wtrace_02�
__inference__destroyer_324768�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zwtrace_0
�
xtrace_02�
__inference__creator_324773�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zxtrace_0
�
ytrace_02�
__inference__initializer_324778�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zytrace_0
�
ztrace_02�
__inference__destroyer_324783�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zztrace_0
.
n0
o1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
:  (2total
:  (2count
.
r0
s1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
�B�
__inference__creator_324755"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__initializer_324763"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__destroyer_324768"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__creator_324773"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__initializer_324778"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__destroyer_324783"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
/:-���2Nadam/embedding/embeddings/m
':%	� 2Nadam/dense_2/kernel/m
 : 2Nadam/dense_2/bias/m
&:$ 2Nadam/dense_3/kernel/m
 :2Nadam/dense_3/bias/m
/:-���2Nadam/embedding/embeddings/v
':%	� 2Nadam/dense_2/kernel/v
 : 2Nadam/dense_2/bias/v
&:$ 2Nadam/dense_3/kernel/v
 :2Nadam/dense_3/bias/v
�B�
__inference_save_fn_324802checkpoint_key"�
���
FullArgSpec
args�
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�	
� 
�B�
__inference_restore_fn_324810restored_tensors_0restored_tensors_1"�
���
FullArgSpec
args� 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
	�	
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant7
__inference__creator_324755�

� 
� "� 7
__inference__creator_324773�

� 
� "� 9
__inference__destroyer_324768�

� 
� "� 9
__inference__destroyer_324783�

� 
� "� B
__inference__initializer_324763D���

� 
� "� ;
__inference__initializer_324778�

� 
� "� �
!__inference__wrapped_model_324027�D���%&-.A�>
7�4
2�/
text_vectorization_input���������
� "1�.
,
dense_3!�
dense_3���������o
__inference_adapt_step_323250NE�C�@
9�6
4�1�
����������IteratorSpec 
� "
 �
C__inference_dense_2_layer_call_and_return_conditional_losses_324730]%&0�-
&�#
!�
inputs����������
� "%�"
�
0��������� 
� |
(__inference_dense_2_layer_call_fn_324719P%&0�-
&�#
!�
inputs����������
� "���������� �
C__inference_dense_3_layer_call_and_return_conditional_losses_324750\-./�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� {
(__inference_dense_3_layer_call_fn_324739O-./�,
%�"
 �
inputs��������� 
� "�����������
E__inference_embedding_layer_call_and_return_conditional_losses_324699b0�-
&�#
!�
inputs����������	
� "+�(
!�
0�����������
� �
*__inference_embedding_layer_call_fn_324690U0�-
&�#
!�
inputs����������	
� "�������������
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_324710{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
;__inference_global_average_pooling1d_1_layer_call_fn_324704nI�F
?�<
6�3
inputs'���������������������������

 
� "!�������������������z
__inference_restore_fn_324810YEK�H
A�>
�
restored_tensors_0
�
restored_tensors_1	
� "� �
__inference_save_fn_324802�E&�#
�
�
checkpoint_key 
� "���
`�]

name�
0/name 
#

slice_spec�
0/slice_spec 

tensor�
0/tensor
`�]

name�
1/name 
#

slice_spec�
1/slice_spec 

tensor�
1/tensor	�
H__inference_sequential_1_layer_call_and_return_conditional_losses_324394�D���%&-.I�F
?�<
2�/
text_vectorization_input���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_324460�D���%&-.I�F
?�<
2�/
text_vectorization_input���������
p

 
� "%�"
�
0���������
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_324610nD���%&-.7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_324683nD���%&-.7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
-__inference_sequential_1_layer_call_fn_324165sD���%&-.I�F
?�<
2�/
text_vectorization_input���������
p 

 
� "�����������
-__inference_sequential_1_layer_call_fn_324328sD���%&-.I�F
?�<
2�/
text_vectorization_input���������
p

 
� "�����������
-__inference_sequential_1_layer_call_fn_324514aD���%&-.7�4
-�*
 �
inputs���������
p 

 
� "�����������
-__inference_sequential_1_layer_call_fn_324537aD���%&-.7�4
-�*
 �
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_324491�D���%&-.]�Z
� 
S�P
N
text_vectorization_input2�/
text_vectorization_input���������"1�.
,
dense_3!�
dense_3���������