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
import androidx.core.app.NotificationManagerCompat
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

    companion object {
        const val CHANNEL_ID = "sensor_channel"
        const val NOTIF_ID = 1
        const val ACTION_START_RECORDING = "com.example.motionpositionsensors.action.START_RECORDING"
        const val ACTION_STOP_RECORDING = "com.example.motionpositionsensors.action.STOP_RECORDING"
        const val ACTION_SENSOR_UPDATE = "com.example.motionpositionsensors.action.SENSOR_UPDATE"
        const val ACTION_RECORDING_SAVED = "com.example.motionpositionsensors.action.RECORDING_SAVED"

        // How often to refresh notification (ms)
        const val NOTIF_UPDATE_INTERVAL_MS = 1000L
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

        // Start initial foreground notification
        val initial = "Sensors active — waiting for data"
        startForeground(
            NOTIF_ID,
            buildNotification("Motion Sensors Active", initial, initial)
        )

        Log.d(TAG, "Foreground service started with initial notification")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        try {
            when (intent?.action) {
                ACTION_START_RECORDING -> {
                    val zoneName = intent.getStringExtra("zoneName") ?: "ZONE"
                    startRecording(zoneName)
                }
                ACTION_STOP_RECORDING -> {
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

            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> acc = event.values.clone()
                Sensor.TYPE_GYROSCOPE -> gyro = event.values.clone()
                Sensor.TYPE_ROTATION_VECTOR -> rotv = event.values.clone()
                Sensor.TYPE_MAGNETIC_FIELD -> mag = event.values.clone()
            }

            // Send broadcast so MainActivity can update its dialog
            val broadcast = Intent(ACTION_SENSOR_UPDATE).apply {
                putExtra("timestamp", ts)
                putExtra("acc", acc)
                putExtra("gyro", gyro)
                putExtra("rotv", rotv)
                putExtra("mag", mag)
                setPackage(packageName)
            }
            sendBroadcast(broadcast)

            // Write to CSV if recording
            if (isRecording) {
                csvWriter?.append(
                    "$ts," +
                            "${safe(acc, 0)},${safe(acc, 1)},${safe(acc, 2)}," +
                            "${safe(gyro, 0)},${safe(gyro, 1)},${safe(gyro, 2)}," +
                            "${safe(rotv, 0)},${safe(rotv, 1)},${safe(rotv, 2)}," +
                            "${safe(mag, 0)},${safe(mag, 1)},${safe(mag, 2)}\n"
                )
            }

            // Update notification every 1s
            val now = System.currentTimeMillis()
            if (now - lastNotifUpdateMs >= NOTIF_UPDATE_INTERVAL_MS) {
                lastNotifUpdateMs = now

                val shortSummary = String.format(
                    "ACC: %.2f,%.2f,%.2f | GYRO: %.2f,%.2f,%.2f",
                    safe(acc, 0), safe(acc, 1), safe(acc, 2),
                    safe(gyro, 0), safe(gyro, 1), safe(gyro, 2)
                )

                val bigText = buildString {
                    append("ACC: X=${fmt(acc, 0)}, Y=${fmt(acc, 1)}, Z=${fmt(acc, 2)}\n")
                    append("GYRO: X=${fmt(gyro, 0)}, Y=${fmt(gyro, 1)}, Z=${fmt(gyro, 2)}\n")
                    append("ROT: X=${fmt(rotv, 0)}, Y=${fmt(rotv, 1)}, Z=${fmt(rotv, 2)}\n")
                    append("MAG: X=${fmt(mag, 0)}, Y=${fmt(mag, 1)}, Z=${fmt(mag, 2)}")
                }

                val title = if (isRecording && currentZoneName != null)
                    "Recording ${currentZoneName}"
                else
                    "Motion Sensors Active"

                val notif = buildNotification(title, shortSummary, bigText)

                // 👇 This is the key change — ensures Android redraws it
                startForeground(NOTIF_ID, notif)

                Log.v(TAG, "Notification updated: $shortSummary")
            }
        } catch (t: Throwable) {
            Log.e(TAG, "onSensorChanged error", t)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun startRecording(zoneName: String) {
        try {
            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val fileName = "${zoneName}_$ts.csv"
            val dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            dir?.let { if (!it.exists()) it.mkdirs() }

            csvFile = File(dir, fileName)
            csvWriter = FileWriter(csvFile!!)
            csvWriter?.append("timestamp,accX,accY,accZ,gyroX,gyroY,gyroZ,rotX,rotY,rotZ,magX,magY,magZ\n")

            isRecording = true
            currentZoneName = zoneName

            val notif = buildNotification("Recording $zoneName", "Recording started", buildSmallBigText())
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
            if (!isRecording) return

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

            val notif = buildNotification(
                "Sensors Active — Not Recording",
                "Recording saved",
                buildSmallBigText()
            )
            startForeground(NOTIF_ID, notif)

            Log.i(TAG, "stopRecording saved=${csvFile?.absolutePath}")
            csvFile = null
            currentZoneName = null
        } catch (e: IOException) {
            Log.e(TAG, "stopRecording failed", e)
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
                NotificationManager.IMPORTANCE_DEFAULT // was LOW
            ).apply {
                description = "Shows live motion sensor data updates"
            }
            nm?.createNotificationChannel(channel)
        }
    }

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
        // 👇 Removed .setOngoing(true) to allow visible updates
        //.setOngoing(true)

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

    private fun safe(arr: FloatArray, idx: Int): Float {
        return if (arr.size > idx && !arr[idx].isNaN()) arr[idx] else 0f
    }

    private fun fmt(arr: FloatArray, idx: Int): String {
        return if (arr.size > idx && !arr[idx].isNaN()) String.format("%.3f", arr[idx]) else "—"
    }
}
