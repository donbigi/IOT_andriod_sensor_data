package com.example.motionpositionsensors

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Environment
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.Date
import kotlin.math.min

class SensorService : Service(), SensorEventListener {

    private val TAG = "SensorService"

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var rotationVector: Sensor? = null
    private var magneticField: Sensor? = null

    // latest sensor values
    private var acc = FloatArray(3) { Float.NaN }
    private var gyro = FloatArray(3) { Float.NaN }
    private var rotv = FloatArray(3) { Float.NaN }
    private var mag = FloatArray(3) { Float.NaN }

    private var isRecording = false
    private var csvWriter: FileWriter? = null
    private var csvFile: File? = null
    private var currentZoneName: String? = null

    // how many post-click samples we've written (ACC-triggered rows)
    private var samplesWritten = 0

    // one sample = one accelerometer event row (with synchronized gyro/rot/mag)
    private data class Sample(
        val tsMs: Long,
        val acc: FloatArray,
        val gyro: FloatArray,
        val rotv: FloatArray,
        val mag: FloatArray
    )

    // Pre-record buffer: keep last PRE_RECORD_MS of samples (ACC-triggered)
    private val preBuffer: ArrayDeque<Sample> = ArrayDeque()

    companion object {
        const val CHANNEL_ID = "sensor_channel"
        const val NOTIF_ID = 1

        const val ACTION_START_RECORDING =
            "com.example.motionpositionsensors.action.START_RECORDING"
        const val ACTION_STOP_RECORDING =
            "com.example.motionpositionsensors.action.STOP_RECORDING"
        const val ACTION_SENSOR_UPDATE =
            "com.example.motionpositionsensors.action.SENSOR_UPDATE"
        const val ACTION_RECORDING_SAVED =
            "com.example.motionpositionsensors.action.RECORDING_SAVED"

        // Notification update throttle
        const val NOTIF_UPDATE_INTERVAL_MS = 1000L

        // How much history (in ms) to keep before click
        const val PRE_RECORD_MS = 600L

        // How many ACC-triggered samples to capture after click
        const val POST_SAMPLES = 400
    }

    private var lastNotifUpdateMs: Long = 0L

    // ------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "onCreate")

        try {
            sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        } catch (e: Exception) {
            Log.e(TAG, "SensorManager init failed", e)
            stopSelf()
            return
        }

        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        magneticField = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        // Use the fastest possible rate for consistent sampling
        val rate = SensorManager.SENSOR_DELAY_FASTEST

        accelerometer?.let { sensorManager.registerListener(this, it, rate) }
        gyroscope?.let { sensorManager.registerListener(this, it, rate) }
        rotationVector?.let { sensorManager.registerListener(this, it, rate) }
        magneticField?.let { sensorManager.registerListener(this, it, rate) }

        createNotificationChannel()
        startForeground(
            NOTIF_ID,
            buildNotification("Sensors Active", "Waiting for data…")
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        try {
            when (intent?.action) {
                ACTION_START_RECORDING -> {
                    val zoneName = intent.getStringExtra("zoneName") ?: "ZONE"
                    Log.d(TAG, "ACTION_START_RECORDING: $zoneName")
                    startRecording(zoneName)
                }
                ACTION_STOP_RECORDING -> {
                    Log.d(TAG, "ACTION_STOP_RECORDING")
                    stopRecording()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "onStartCommand error", e)
        }
        // We just want the service to keep running
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            sensorManager.unregisterListener(this)
        } catch (_: Exception) {
        }
        stopRecording()
        Log.d(TAG, "onDestroy")
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // ------------------------------------------------------------
    // Sensor callbacks
    // ------------------------------------------------------------

    override fun onSensorChanged(event: SensorEvent) {
        try {
            // Use monotonic sensor time, not wall clock
            val tsMs = event.timestamp / 1_000_000L

            // Update the latest values for the specific sensor
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> acc = event.values.clone()
                Sensor.TYPE_GYROSCOPE -> gyro = event.values.clone()
                Sensor.TYPE_ROTATION_VECTOR -> {
                    // Some devices give 4D rotation vector; we only keep first 3
                    val arr = event.values
                    rotv = FloatArray(3) {
                        if (it < arr.size) arr[it] else Float.NaN
                    }
                }
                Sensor.TYPE_MAGNETIC_FIELD -> mag = event.values.clone()
            }

            // Snapshot all current values at this timestamp
            val sample = Sample(
                tsMs = tsMs,
                acc = acc.clone(),
                gyro = gyro.clone(),
                rotv = rotv.clone(),
                mag = mag.clone()
            )

            // Broadcast to UI for live display (can be all events, doesn't affect CSV)
            Intent(ACTION_SENSOR_UPDATE).apply {
                putExtra("acc", sample.acc)
                putExtra("gyro", sample.gyro)
                putExtra("rotv", sample.rotv)
                putExtra("mag", sample.mag)
                setPackage(packageName)
            }.also { sendBroadcast(it) }

            // Only ACC events define a "row" in the time series
            if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {

                if (isRecording) {
                    // Write post-click sample
                    writeSampleToCsv(sample)
                    samplesWritten++

                    if (samplesWritten >= POST_SAMPLES) {
                        Log.i(TAG, "Reached POST_SAMPLES = $POST_SAMPLES, stopping recording")
                        stopRecording()
                    }

                } else {
                    // Not recording: maintain pre-buffer window based on sensor time
                    synchronized(preBuffer) {
                        preBuffer.addLast(sample)

                        // Drop anything older than PRE_RECORD_MS
                        while (preBuffer.isNotEmpty()) {
                            val oldest = preBuffer.first()
                            if (sample.tsMs - oldest.tsMs > PRE_RECORD_MS) {
                                preBuffer.removeFirst()
                            } else {
                                break
                            }
                        }
                    }
                }
            }

            // Throttle notification updates by wall clock
            // Throttle notification updates by wall clock
            val nowWall = System.currentTimeMillis()
            if (nowWall - lastNotifUpdateMs >= NOTIF_UPDATE_INTERVAL_MS) {
                lastNotifUpdateMs = nowWall

                val title = if (isRecording && currentZoneName != null) {
                    "Recording $currentZoneName"
                } else {
                    "Sensors Active"
                }

                // Short single-line summary
                val content = "ACC: %.2f %.2f %.2f".format(
                    safe(acc, 0),
                    safe(acc, 1),
                    safe(acc, 2)
                )

                // Multiline detailed view (shows when you expand)
                val bigText = """
                    ACC: %.2f %.2f %.2f
                    GYRO: %.2f %.2f %.2f
                    ROT : %.2f %.2f %.2f
                    MAG : %.2f %.2f %.2f
                """.trimIndent().format(
                    safe(acc, 0),  safe(acc, 1),  safe(acc, 2),
                    safe(gyro, 0), safe(gyro, 1), safe(gyro, 2),
                    safe(rotv, 0), safe(rotv, 1), safe(rotv, 2),
                    safe(mag, 0),  safe(mag, 1),  safe(mag, 2)
                )

                startForeground(NOTIF_ID, buildNotification(title, content, bigText))
            }


        } catch (e: Exception) {
            Log.e(TAG, "onSensorChanged error", e)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // no-op
    }

    // ------------------------------------------------------------
    // Recording logic
    // ------------------------------------------------------------

    private fun startRecording(zoneName: String) {
        try {
            // Prepare CSV
            val tsStr = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val fileName = "${zoneName}_$tsStr.csv"

            val dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            if (dir != null && !dir.exists()) dir.mkdirs()

            csvFile = File(dir, fileName)
            csvWriter = FileWriter(csvFile!!)

            csvWriter!!.append(
                "timestamp,accX,accY,accZ," +
                        "gyroX,gyroY,gyroZ," +
                        "rotX,rotY,rotZ," +
                        "magX,magY,magZ\n"
            )

            // Flush pre-buffer contents (1 second BEFORE tap)
            synchronized(preBuffer) {
                if (preBuffer.isNotEmpty()) {
                    Log.i(TAG, "Flushing ${preBuffer.size} pre-buffer samples to CSV")
                    for (s in preBuffer) {
                        writeSampleToCsv(s)
                    }
                    preBuffer.clear()
                } else {
                    Log.i(TAG, "No pre-buffer samples to flush")
                }
            }

            isRecording = true
            currentZoneName = zoneName
            samplesWritten = 0

            startForeground(
                NOTIF_ID,
                buildNotification("Recording $zoneName", "Capturing motion…")
            )

            Log.i(TAG, "Recording started: ${csvFile?.absolutePath}")

        } catch (e: Exception) {
            Log.e(TAG, "startRecording error", e)
            isRecording = false
            currentZoneName = null
            samplesWritten = 0
        }
    }

    private fun stopRecording() {
        try {
            if (!isRecording) {
                Log.w(TAG, "stopRecording called but isRecording=false")
                return
            }

            isRecording = false

            csvWriter?.flush()
            csvWriter?.close()
            csvWriter = null

            // Notify UI / app that file is saved
            Intent(ACTION_RECORDING_SAVED).apply {
                putExtra("fileName", csvFile?.name)
                putExtra("filePath", csvFile?.absolutePath)
                setPackage(packageName)
            }.also { sendBroadcast(it) }

            Log.i(TAG, "Recording saved: ${csvFile?.absolutePath}")

            csvFile = null
            currentZoneName = null
            samplesWritten = 0

            // Back to idle notification
            startForeground(
                NOTIF_ID,
                buildNotification("Sensors Active", "Recording saved")
            )

        } catch (e: IOException) {
            Log.e(TAG, "stopRecording IO error", e)
        } catch (e: Exception) {
            Log.e(TAG, "stopRecording error", e)
        }
    }

    private fun writeSampleToCsv(s: Sample) {
        try {
            csvWriter?.append(
                "${s.tsMs}," +
                        "${safe(s.acc, 0)},${safe(s.acc, 1)},${safe(s.acc, 2)}," +
                        "${safe(s.gyro, 0)},${safe(s.gyro, 1)},${safe(s.gyro, 2)}," +
                        "${safe(s.rotv, 0)},${safe(s.rotv, 1)},${safe(s.rotv, 2)}," +
                        "${safe(s.mag, 0)},${safe(s.mag, 1)},${safe(s.mag, 2)}\n"
            )
        } catch (e: IOException) {
            Log.e(TAG, "CSV write error", e)
        }
    }

    // ------------------------------------------------------------
    // Notification helpers
    // ------------------------------------------------------------

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Motion Sensor Service",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Capturing motion sensor data"
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm?.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(
        title: String,
        content: String,
        bigText: String? = null
    ): Notification {
        val openIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pending = PendingIntent.getActivity(
            this,
            0,
            openIntent,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S)
                PendingIntent.FLAG_IMMUTABLE
            else
                0
        )

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentTitle(title)
            .setContentText(content)
            .setContentIntent(pending)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setOngoing(true)

        if (bigText != null) {
            builder.setStyle(
                NotificationCompat.BigTextStyle().bigText(bigText)
            )
        }

        return builder.build()
    }


    // ------------------------------------------------------------
    // Small helpers
    // ------------------------------------------------------------

    private fun safe(arr: FloatArray, idx: Int): Float {
        return if (idx in arr.indices && !arr[idx].isNaN()) arr[idx] else 0f
    }
}
