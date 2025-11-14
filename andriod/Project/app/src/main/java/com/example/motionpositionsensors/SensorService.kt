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
import java.util.*

class SensorService : Service(), SensorEventListener {
    private val TAG = "SensorService"

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var rotationVector: Sensor? = null
    private var magneticField: Sensor? = null

    private var acc = FloatArray(3) { Float.NaN }
    private var gyro = FloatArray(3) { Float.NaN }
    private var rotv = FloatArray(3) { Float.NaN }
    private var mag = FloatArray(3) { Float.NaN }

    private var isRecording = false
    private var csvWriter: FileWriter? = null
    private var csvFile: File? = null
    private var currentZoneName: String? = null

    // --- Pre-record buffer: keep recent sensor snapshots for pre-click context ---
    private data class Sample(
        val ts: Long,
        val acc: FloatArray,
        val gyro: FloatArray,
        val rotv: FloatArray,
        val mag: FloatArray
    )

    // buffer (deque) storing Sample in chronological order (oldest -> newest)
    private val preBuffer: ArrayDeque<Sample> = ArrayDeque()

    // 🔥 NEW: auto-stop deadline for post-click recording (wall-clock ms)
    private var postEndTime: Long = 0L

    companion object {
        const val CHANNEL_ID = "sensor_channel"
        const val NOTIF_ID = 1
        const val ACTION_START_RECORDING = "com.example.motionpositionsensors.action.START_RECORDING"
        const val ACTION_STOP_RECORDING = "com.example.motionpositionsensors.action.STOP_RECORDING"
        const val ACTION_SENSOR_UPDATE = "com.example.motionpositionsensors.action.SENSOR_UPDATE"
        const val ACTION_RECORDING_SAVED = "com.example.motionpositionsensors.action.RECORDING_SAVED"

        // How often to refresh notification (ms)
        const val NOTIF_UPDATE_INTERVAL_MS = 1000L

        // PRE-RECORD WINDOW: how many milliseconds of data to keep before a click
        // Keep at 1000 ms (1 second). Change this if you want shorter/longer.
        const val PRE_RECORD_MS = 1000L
    }

    private var lastNotifUpdateMs = 0L

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "onCreate")
        try {
            sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        } catch (t: Throwable) {
            Log.e(TAG, "getSystemService failed", t)
            stopSelf()
            return
        }

        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        magneticField = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        val rate = SensorManager.SENSOR_DELAY_GAME
        accelerometer?.let { sensorManager.registerListener(this, it, rate) }
        gyroscope?.let { sensorManager.registerListener(this, it, rate) }
        rotationVector?.let { sensorManager.registerListener(this, it, rate) }
        magneticField?.let { sensorManager.registerListener(this, it, rate) }

        createNotificationChannel()

        // Start initial foreground notification (keeps behavior identical to last working version)
        val initial = "Sensors active — waiting for data"
        try {
            startForeground(
                NOTIF_ID,
                buildNotification("Motion Sensors Active", initial, initial)
            )
            Log.d(TAG, "Foreground service started with initial notification")
        } catch (sec: SecurityException) {
            Log.e(TAG, "startForeground SecurityException", sec)
            stopSelf()
            return
        } catch (t: Throwable) {
            Log.e(TAG, "startForeground unexpected", t)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        try {
            when (intent?.action) {
                ACTION_START_RECORDING -> {
                    val zoneName = intent.getStringExtra("zoneName") ?: "ZONE"
                    Log.d(TAG, "onStartCommand: START_RECORDING -> $zoneName")
                    startRecording(zoneName)
                }
                ACTION_STOP_RECORDING -> {
                    Log.d(TAG, "onStartCommand: STOP_RECORDING")
                    stopRecording()
                }
                else -> Log.d(TAG, "onStartCommand: action=${intent?.action}")
            }
        } catch (t: Throwable) {
            Log.e(TAG, "onStartCommand error", t)
        }
        return START_STICKY
    }

    override fun onSensorChanged(event: SensorEvent) {
        try {
            val ts = System.currentTimeMillis()

            // update snapshot for the sensor that fired
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> acc = event.values.clone()
                Sensor.TYPE_GYROSCOPE -> gyro = event.values.clone()
                Sensor.TYPE_ROTATION_VECTOR -> rotv = event.values.clone()
                Sensor.TYPE_MAGNETIC_FIELD -> mag = event.values.clone()
            }

            // Take a snapshot of all sensors now
            val snapAcc = acc.clone()
            val snapGyro = gyro.clone()
            val snapRotv = rotv.clone()
            val snapMag = mag.clone()
            val sample = Sample(ts, snapAcc, snapGyro, snapRotv, snapMag)

            // broadcast updated arrays (scoped to our app)
            val b = Intent(ACTION_SENSOR_UPDATE).apply {
                putExtra("timestamp", ts)
                putExtra("acc", snapAcc)
                putExtra("gyro", snapGyro)
                putExtra("rotv", snapRotv)
                putExtra("mag", snapMag)
                setPackage(packageName)
            }
            sendBroadcast(b)

            // If currently recording -> append immediately to CSV
            if (isRecording) {
                writeSampleToCsv(sample)

                // 🔥 NEW: auto-stop 1.5s after startRecording()
                if (System.currentTimeMillis() >= postEndTime) {
                    Log.i(TAG, "Auto-stop reached at postEndTime=$postEndTime")
                    stopRecording()
                }
            } else {
                // not recording: add to pre-buffer and prune older samples beyond PRE_RECORD_MS
                synchronized(preBuffer) {
                    preBuffer.addLast(sample)

                    while (true) {
                        val first = preBuffer.firstOrNull() ?: break
                        if (ts - first.ts > PRE_RECORD_MS) {
                            preBuffer.removeFirst()
                        } else {
                            break
                        }
                    }
                }
            }

            // throttle notification updates (same behavior as last working version)
            val now = System.currentTimeMillis()
            if (now - lastNotifUpdateMs >= NOTIF_UPDATE_INTERVAL_MS) {
                lastNotifUpdateMs = now

                val shortSummary = String.format(
                    "ACC: %.2f,%.2f,%.2f | GYRO: %.2f,%.2f,%.2f",
                    safe(snapAcc, 0), safe(snapAcc, 1), safe(snapAcc, 2),
                    safe(snapGyro, 0), safe(snapGyro, 1), safe(snapGyro, 2)
                )

                val bigText = buildString {
                    append("ACC: X=${fmt(snapAcc, 0)}, Y=${fmt(snapAcc, 1)}, Z=${fmt(snapAcc, 2)}\n")
                    append("GYRO: X=${fmt(snapGyro, 0)}, Y=${fmt(snapGyro, 1)}, Z=${fmt(snapGyro, 2)}\n")
                    append("ROT: X=${fmt(snapRotv, 0)}, Y=${fmt(snapRotv, 1)}, Z=${fmt(snapRotv, 2)}\n")
                    append("MAG: X=${fmt(snapMag, 0)}, Y=${fmt(snapMag, 1)}, Z=${fmt(snapMag, 2)}")
                }

                val title = if (isRecording && currentZoneName != null) "Recording $currentZoneName" else "Motion Sensors Active"

                val notif = buildNotification(title, shortSummary, bigText)

                // ensure Android redraws the foreground notification (exactly like last working)
                startForeground(NOTIF_ID, notif)

                Log.v(TAG, "Notification updated: $shortSummary")
            }
        } catch (t: Throwable) {
            Log.e(TAG, "onSensorChanged error", t)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // no-op
    }

    private fun startRecording(zoneName: String) {
        try {
            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val fileName = "${zoneName}_$ts.csv"
            val dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            dir?.let { if (!it.exists()) it.mkdirs() }

            csvFile = File(dir, fileName)
            csvWriter = FileWriter(csvFile!!)
            csvWriter?.append("timestamp,accX,accY,accZ,gyroX,gyroY,gyroZ,rotX,rotY,rotZ,magX,magY,magZ\n")

            // First: write a snapshot of the pre-buffer contents (chronological order)
            synchronized(preBuffer) {
                if (preBuffer.isNotEmpty()) {
                    val toFlush = preBuffer.toList() // snapshot to avoid concurrent modification
                    Log.i(TAG, "Flushing ${toFlush.size} pre-record samples to CSV for $zoneName")
                    for (s in toFlush) {
                        writeSampleToCsv(s)
                    }
                    preBuffer.clear()
                } else {
                    Log.i(TAG, "No pre-record samples available to flush")
                }
            }

            isRecording = true
            currentZoneName = zoneName

            // 🔥 NEW: set auto-stop deadline: now + 1500 ms (1.5 seconds)
            postEndTime = System.currentTimeMillis() + 1500L

            val notif = buildNotification("Recording $zoneName", "Recording started", buildSmallBigText())
            // keep notification visible and updated as foreground
            startForeground(NOTIF_ID, notif)

            Log.i(TAG, "startRecording -> ${csvFile?.absolutePath}")
        } catch (e: IOException) {
            Log.e(TAG, "startRecording failed", e)
            isRecording = false
            currentZoneName = null
        }
    }

    private fun stopRecording() {
        try {
            if (!isRecording) {
                Log.w(TAG, "stopRecording called but not recording")
                return
            }
            isRecording = false
            csvWriter?.flush()
            csvWriter?.close()
            csvWriter = null

            val saved = Intent(ACTION_RECORDING_SAVED).apply {
                putExtra("filePath", csvFile?.absolutePath)
                putExtra("fileName", csvFile?.name)
                setPackage(packageName)
            }
            sendBroadcast(saved)
            Log.i(TAG, "stopRecording saved=${csvFile?.absolutePath}")
            csvFile = null
            currentZoneName = null

            // notify user
            val notif = buildNotification("Sensors active — not recording", "Recording saved", buildSmallBigText())
            startForeground(NOTIF_ID, notif)
        } catch (e: IOException) {
            Log.e(TAG, "stopRecording failed", e)
        }
    }

    // helper to write a Sample into the CSV writer (safely)
    private fun writeSampleToCsv(s: Sample) {
        try {
            csvWriter?.append(
                "${s.ts}," +
                        "${safe(s.acc, 0)},${safe(s.acc, 1)},${safe(s.acc, 2)}," +
                        "${safe(s.gyro, 0)},${safe(s.gyro, 1)},${safe(s.gyro, 2)}," +
                        "${safe(s.rotv, 0)},${safe(s.rotv, 1)},${safe(s.rotv, 2)}," +
                        "${safe(s.mag, 0)},${safe(s.mag, 1)},${safe(s.mag, 2)}\n"
            )
        } catch (e: IOException) {
            Log.e(TAG, "writeSampleToCsv failed", e)
        }
    }

    private fun buildSmallBigText(): String {
        return buildString {
            append("ACC: ${fmt(acc, 0)},${fmt(acc, 1)},${fmt(acc, 2)}\n")
            append("GYRO: ${fmt(gyro, 0)},${fmt(gyro, 1)},${fmt(gyro, 2)}")
        }
    }

    private fun createNotificationChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Sensor Background",
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = "Shows live motion sensor data updates"
            }
            nm?.createNotificationChannel(channel)
        }
    }

    /**
     * Build a notification. If bigText is provided, use BigTextStyle to show multi-line content.
     */
    private fun buildNotification(title: String, content: String, bigText: String? = null): Notification {
        val openIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pending = PendingIntent.getActivity(
            this,
            0,
            openIntent,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) PendingIntent.FLAG_IMMUTABLE else 0
        )

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(content)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentIntent(pending)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(false)

        if (!bigText.isNullOrEmpty()) {
            builder.setStyle(NotificationCompat.BigTextStyle().bigText(bigText))
        }
        return builder.build()
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            sensorManager.unregisterListener(this)
        } catch (t: Throwable) {
            Log.w(TAG, "unregister listener failed", t)
        }
        stopRecording()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // --- small helpers ----
    private fun safe(arr: FloatArray, idx: Int): Float {
        return if (arr.size > idx && !arr[idx].isNaN()) arr[idx] else 0f
    }

    private fun fmt(arr: FloatArray, idx: Int): String {
        return if (arr.size > idx && !arr[idx].isNaN()) String.format("%.3f", arr[idx]) else "—"
    }
}
