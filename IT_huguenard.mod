TITLE Low threshold calcium current
:
:   Ca++ current responsible for low threshold spikes (LTS)
:   THALAMOCORTICAL CELLS
:   Differential equations
:
:   Model based on the data of Huguenard & McCormick, J Neurophysiol
:   68: 1373-1383, 1992 and Huguenard & Prince, J Neurosci.
:   12: 3804-3817, 1992.
:
:   Features:
:
:	- kinetics described by Nernst equations using a m2h format
:	- activation considered at steady-state
:	- inactivation fit to Huguenard's data using a bi-exp function
:	- shift for screening charge, q10 of inactivation of 3
:
:
:   Written by Alain Destexhe, Salk Institute, 1993; modified 1995
:

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX it
	USEION ca READ cai,cao WRITE ica
	GLOBAL q10
	RANGE gcabar, m_inf, tau_m, h_inf, tau_h, shift
}

UNITS {
	(molar) = (1/liter)
	(mV) =	(millivolt)
	(mA) =	(milliamp)
	(mM) =	(millimolar)

	FARADAY = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
}

PARAMETER {
	v		(mV)
	celsius	= 36	(degC)
	gcabar	= 0.002	(mho/cm2)
	q10	= 3			: Q10 of inactivation
	shift	= 2 	(mV)		: corresponds to 2mM ext Ca++
	cai	= 2.4e-4 (mM)		: adjusted for eca=120 mV
	cao	= 2	(mM)
}

STATE {
	h
}

ASSIGNED {
	ica	(mA/cm2)
	carev	(mV)
	m_inf
	tau_m	(ms)			: dummy variable for compatibility
	h_inf
	tau_h	(ms)
	phi_h
}

BREAKPOINT {
	SOLVE castate METHOD cnexp
	carev = (1e3) * (R*(celsius+273.15))/(2*FARADAY) * log (cao/cai)
	ica = gcabar * m_inf * m_inf * h * (v-carev)
}

DERIVATIVE castate {
	evaluate_fct(v)

	h' = (h_inf - h) / tau_h
}


UNITSOFF
INITIAL {
	h = 0

:
:   Transformation to 36 deg assuming Q10 of 3 for h
:   (as in Coulter et al., J Physiol 414: 587, 1989)
:
	phi_h = q10 ^ ((celsius-24 (degC) )/10 (degC) )
}

PROCEDURE evaluate_fct(v(mV)) { LOCAL Vm

	Vm = v + shift

	m_inf = 1.0 / ( 1 + exp(-(Vm+57)/6.2) )
	h_inf = 1.0 / ( 1 + exp((Vm+81)/4.0) )

:	if(Vm < -80) {
:		tau_h = exp((Vm+467)/66.6) / phi_h
:	} else {
:		tau_h = ( 28 + exp(-(Vm+22)/10.5) ) / phi_h
:	}

	tau_h = 30.8 + (211.4 + exp((Vm+113.2)/5)) / (1 + exp((Vm+84)/3.2))

	tau_h = tau_h / phi_h

}

UNITSON
