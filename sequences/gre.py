
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.opts import Opts
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_adc import make_adc

# Sistema MRI
system = Opts(
    max_grad=28,
    grad_unit='mT/m',
    max_slew=150,
    slew_unit='T/m/s',
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6
)

seq = Sequence(system)

# Parámetros
flip_angle = 15 * np.pi / 180
slice_thickness = 5e-3
rf_duration = 3e-3
Nx = 256
adc_raster = system.adc_raster_time
adc_dwell = 10 * adc_raster   # dwell válido (ej: 10 raster ticks)
grad_raster = system.grad_raster_time
readout_time = np.ceil((Nx * adc_dwell) / grad_raster) * grad_raster

# RF + slice
rf, gz, gz_reph = make_sinc_pulse(
    flip_angle=flip_angle,
    duration=rf_duration,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system,
    return_gz=True,
    #rf_delay=system.rf_dead_time
)


# Gradiente de lectura
gx = make_trapezoid(
    channel='x',
    flat_area=Nx,
    flat_time=readout_time,
    system=system
)

# ADC (centrado en el gradiente)
#adc_dwell = gx.flat_time / Nx

adc = make_adc(
    num_samples=Nx,
    dwell=adc_dwell,
    delay=gx.rise_time,
    system=system
)


# Construcción de la secuencia
seq.add_block(rf, gz)
seq.add_block(gz_reph)
seq.add_block(gx, adc)

print("GRE completa creada")

# Visualización
seq.plot()

# Validación del timing
ok, error_report = seq.check_timing()
if ok:
    print("Timing OK")
else:
    print("Timing ERROR")
    print(error_report)

# Guardar secuencia
seq.write(r"C:\Users\ihealth\gre_basic.seq")
print("Archivo gre_basic.seq guardado")
