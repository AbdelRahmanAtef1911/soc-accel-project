/*
 * sensor_regs.h - register map of the course-project streaming sensor
 * (accel/sensor_dma.h). Shared by the platform model and the bare-metal
 * software, like accel_regs.h.
 *
 * The sensor produces one S-byte sample every 1/R seconds into a FIFO of
 * DEPTH samples; its DMA drains THETA samples per bus request (request/grant
 * bus, accel/rg_bus.h). Software only uses the registers to open a
 * measurement window and to read the window's results.
 */
#ifndef SENSOR_REGS_H
#define SENSOR_REGS_H

#define SENSOR_BASE        0x11000000u
#define SENSOR_SIZE        0x00001000u

#define SENSOR_ID          0x00u  /* R: 0x53454E53 ("SENS"), 0 if no sensor */
#define SENSOR_WINDOW_US   0x04u  /* W: length of the measurement window (us) */
#define SENSOR_MARK        0x08u  /* W: any value opens the window now        */
#define SENSOR_OVF_WINDOW  0x0Cu  /* R: samples dropped inside the window     */
#define SENSOR_OVF_TOTAL   0x10u  /* R: samples dropped since reset           */
#define SENSOR_PEAK_WINDOW 0x14u  /* R: highest FIFO occupancy in the window  */
#define SENSOR_THETA       0x18u  /* R: drain threshold (samples)             */
#define SENSOR_DEPTH       0x1Cu  /* R: FIFO depth (samples)                  */
#define SENSOR_RATE_KSPS   0x20u  /* R: sample rate (thousand samples/s)      */
#define SENSOR_MAXWAIT     0x24u  /* R: longest request->grant wait in the window (cycles) */
#define SENSOR_WINDOW_DONE 0x28u  /* R: 1 once the window has closed          */

#define SENSOR_ID_VALUE    0x53454E53u

#endif /* SENSOR_REGS_H */
