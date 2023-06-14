
#################################3
### EDIT
#################################

#Psi1=+1e-4 # forward
#Psi1=0 # forward
#Psi1_orig = -.075 #
#Psi1 = -6.5

#########
Psi1_orig = 0
Psi2_orig = -.3 #reverse
Psi2_orig = .2 #
Psi3_orig = .03
#########
Psi1_orig = 0
Psi2_orig = 5 #reverse: - 
Psi3_orig = 20
Psi3_orig = 0
#########


#########
X_norma = .10 * 1e-6 * 100 # mikron -> m -> cm
#########


################################
# DOPING
################################
# Background P:
Na4_orig = 1e16
#diffusion N:
Nd2_orig =1.3e16


#ostatne nehraju rolu
#drift
Nd3_orig = 1e14 # N  epitax
#C
Nd4_orig = 5e18 # N  substrate
#B
Na2_orig =  2e17 # P  diffusion
#E
Nd1_orig = 9e18 # N diffusion
################################


#######################
ylenC = 2;
ylendrift = 10;
ylenB = 10;
ylenE = 2;
#######################
xlenE = .5;
xlenB = 0;
xlen_medzera = 0;
#######################

yCd = 0+ylenC;
yBC = yCd+ylendrift;
yBE = yBC+ylenB;
ylen = yBE+ylenE;

xBmedzera1 = xlenB;
xBE1 = xBmedzera1+xlen_medzera;
xlen = xBE1 + xlenE;

meshsize = 0.2*ylen;

XLEN = xlen
YLEN = ylen

###########
###########
ni_orig = 1e10
##########



q =  1.602e-19
#eps0 = 8.854e-12 * 1/m2cm # [F / m] * 1/m2cm
eps0 =  8.854e-14 # [F/cm]
epsr_Si = 11.68
eps_Si = eps0 * epsr_Si

Vth = 0.0259 # k.T/q

mob_n_orig = 1*1400 # [cm2 / V / s]
mob_p_orig = 450
D_p_orig = 1*Vth*mob_p_orig
D_n_orig = 1*Vth*mob_n_orig
#D_n = .6# 36 # [cm2 / s]
#D_p = .6# 12
Tau_p_orig = .001e-9 # s
Tau_n_orig = .002e-9 # s
Tau_p_orig = 05e-12 # s
Tau_n_orig = 10e-12 # s

#########

#X_norma = .3e-5 * 100 # m -> cm
Psi_norma = 0.0259 # k.T/q
N_norma = max(Na4_orig, Nd2_orig)
#N_norma = ni_orig
#N_norma = N_norma*1e2
mob_norma = max(mob_p_orig, mob_n_orig)
D_norma = Psi_norma * mob_norma

t_norma = X_norma**2/(Psi_norma*mob_norma)

Na4 =  Na4_orig / N_norma
Nd2 =  Nd2_orig / N_norma
ni = ni_orig / N_norma

#Psi1 =  Psi1_orig / Psi_norma
#Psi2 =  Psi2_orig / Psi_norma
#Psi3 =  Psi3_orig / Psi_norma
Psi1 =  Psi1_orig / 1
Psi2 =  Psi2_orig / 1
Psi3 =  Psi3_orig / 1
mob_p = mob_p_orig / mob_norma
mob_n = mob_n_orig / mob_norma
D_p = Dp = D_p_orig / D_norma
D_n = Dn = D_n_orig / D_norma
Tau_n = Tau_n_orig/t_norma
Tau_p = Tau_p_orig/t_norma
#########

lmda0 = eps_Si * Psi_norma / (X_norma**2 * q * N_norma)
#lmda1 = N_norma * Psi_norma / X_norma**2 * mob_p
#lmda2 = N_norma * Psi_norma / X_norma**2 * mob_n
lmda1 = N_norma * D_norma / X_norma**2
lmda2 = N_norma * D_norma / X_norma**2

lmda_J = lmdaJ = q * N_norma * D_norma / X_norma

#########
