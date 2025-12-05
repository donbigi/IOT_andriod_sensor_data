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

class SensorService : Service(), SensorEventListener {

    private val TAG = "SensorService"

    // ==========================================
    // BACKEND IP & PORT (CONFIGURABLE)
    // ==========================================
    private var backendIp: String = "10.173.39.86"   // you can change at runtime
    private val backendPort: Int = 8000

    private var lastTapTime: Long = 0L
    private val finalizeDelayMs = 3000L  // 5 seconds inactivity

    private fun backendUrl(path: String): String {
        return "http://$backendIp:$backendPort$path"
    }

    // Allow MainActivity to update IP dynamically
    fun updateBackendIp(ip: String) {
        backendIp = ip
        Log.i(TAG, "Backend IP updated → $backendIp")
    }

    // ==========================================
    // Sensors
    // ==========================================
    private var lastStoppedZone: String? = null

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

    private var samplesWritten = 0


    data class Sample(
        val tsMs: Long,
        val acc: FloatArray,
        val gyro: FloatArray,
        val rotv: FloatArray,
        val mag: FloatArray
    )

    private val preBuffer: ArrayDeque<Sample> = ArrayDeque()

    companion object {
        var appInForeground: Boolean = true
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

        const val NOTIF_UPDATE_INTERVAL_MS = 1000L
        const val PRE_RECORD_MS = 600L
        const val POST_SAMPLES = 400
    }

    private var lastNotifUpdateMs: Long = 0L

    // ==========================================
    // Lifecycle
    // ==========================================

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        magneticField = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        val rate = SensorManager.SENSOR_DELAY_FASTEST

        accelerometer?.let { sensorManager.registerListener(this, it, rate) }
        gyroscope?.let { sensorManager.registerListener(this, it, rate) }
        rotationVector?.let { sensorManager.registerListener(this, it, rate) }
        magneticField?.let { sensorManager.registerListener(this, it, rate) }

        createNotificationChannel()
        startForeground(NOTIF_ID, buildNotification("Sensors Active", "Waiting…"))

        Thread {
            while (true) {
                Thread.sleep(1000)

                if (!isRecording) {
                    val now = System.currentTimeMillis()

                    // Only finalize for ZONE-based recordings, not manual
                    if (lastStoppedZone != "manual" &&
                        now - lastTapTime >= finalizeDelayMs
                    ) {
                        sendFinalizeRequest()
                        lastTapTime = Long.MAX_VALUE
                    }
                }

            }
        }.start()

    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_RECORDING -> {
                val zoneName = intent.getStringExtra("zoneName") ?: "ZONE"
                startRecording(zoneName)
            }
            ACTION_STOP_RECORDING -> {
                stopRecording()
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        stopRecording()
    }

    override fun onBind(intent: Intent?): IBinder? = null


    // ==========================================
    // Sensor Data Listener
    // ==========================================

    override fun onSensorChanged(event: SensorEvent) {
        val tsMs = event.timestamp / 1_000_000L

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> acc = event.values.clone()
            Sensor.TYPE_GYROSCOPE -> gyro = event.values.clone()
            Sensor.TYPE_ROTATION_VECTOR -> {
                val v = event.values
                rotv = FloatArray(3) { if (it < v.size) v[it] else 0f }
            }
            Sensor.TYPE_MAGNETIC_FIELD -> mag = event.values.clone()
        }

        val sample = Sample(tsMs, acc.clone(), gyro.clone(), rotv.clone(), mag.clone())

        // Broadcast live sensor reading to UI
        Intent(ACTION_SENSOR_UPDATE).apply {
            putExtra("acc", sample.acc)
            putExtra("gyro", sample.gyro)
            putExtra("rotv", sample.rotv)
            putExtra("mag", sample.mag)
            setPackage(packageName)
        }.also { sendBroadcast(it) }

        // === CSV PRE/POST handling
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {

            if (isRecording) {

                // Only count if app is foreground
                if (SensorService.appInForeground) {
                    writeSampleToCsv(sample)
                    samplesWritten++

                    // Auto-stop for zone recordings only
                    if (currentZoneName != "manual" && samplesWritten >= POST_SAMPLES) {
                        stopRecording()
                    }
                }

            } else {
                synchronized(preBuffer) {
                    preBuffer.addLast(sample)
                    while (preBuffer.isNotEmpty() &&
                        (tsMs - preBuffer.first().tsMs) > PRE_RECORD_MS
                    ) {
                        preBuffer.removeFirst()
                    }
                }
            }
        }


        // Notification throttling
        val nowWall = System.currentTimeMillis()
        if (nowWall - lastNotifUpdateMs >= NOTIF_UPDATE_INTERVAL_MS) {
            lastNotifUpdateMs = nowWall
            val notifText = """
            ACC  → x:%.2f,  y:%.2f,  z:%.2f
            GYRO → x:%.2f,  y:%.2f,  z:%.2f
            ROT  → x:%.2f,  y:%.2f,  z:%.2f
            MAG  → x:%.2f,  y:%.2f,  z:%.2f
            """.trimIndent().format(
                acc[0], acc[1], acc[2],
                gyro[0], gyro[1], gyro[2],
                rotv[0], rotv[1], rotv[2],
                mag[0], mag[1], mag[2]
            )

            startForeground(
                NOTIF_ID,
                buildNotification(
                    if (isRecording) "Recording $currentZoneName" else "Sensors Active",
                    notifText
                )
            )

        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // ==========================================
    // Recording Logic
    // ==========================================

    private fun startRecording(zoneName: String) {
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

        synchronized(preBuffer) {
            for (s in preBuffer) writeSampleToCsv(s)
            preBuffer.clear()
        }

        isRecording = true
        currentZoneName = zoneName
        samplesWritten = 0
        lastTapTime = System.currentTimeMillis()

        Log.i(TAG, "Recording started → ${csvFile?.absolutePath}")
    }

    private fun stopRecording() {
        if (!isRecording) return

        isRecording = false

        csvWriter?.flush()
        csvWriter?.close()

        val savedFile = csvFile
        val zoneAtStop = currentZoneName
        lastStoppedZone = zoneAtStop
        csvFile = null
        samplesWritten = 0
        currentZoneName != "manual"
        lastTapTime = System.currentTimeMillis()


        // Notify UI file saved
        Intent(ACTION_RECORDING_SAVED).apply {
            putExtra("filePath", savedFile?.absolutePath)
            putExtra("fileName", savedFile?.name)
            setPackage(packageName)
        }.also { sendBroadcast(it) }

        // ===============================
        // SEND CSV TO BACKEND (dynamic IP)
        // ===============================
        // SEND CSV TO BACKEND (skip if manual recording)
        // SEND CSV TO BACKEND (skip if manual)
        if (savedFile != null &&
            savedFile.exists() &&
            zoneAtStop != "manual")
        {
            Thread {
                try {
                    val url = java.net.URL(backendUrl("/predict-tap"))
                    val boundary = "----PINBOUNDARY" + System.currentTimeMillis()

                    val conn = url.openConnection() as java.net.HttpURLConnection
                    conn.requestMethod = "POST"
                    conn.doOutput = true
                    conn.setRequestProperty(
                        "Content-Type",
                        "multipart/form-data; boundary=$boundary"
                    )

                    val output = conn.outputStream
                    val writer = output.bufferedWriter()

                    writer.append("--$boundary\r\n")
                    writer.append(
                        "Content-Disposition: form-data; name=\"file\"; filename=\"${savedFile.name}\"\r\n"
                    )
                    writer.append("Content-Type: text/csv\r\n\r\n")
                    writer.flush()

                    savedFile.inputStream().use { it.copyTo(output) }
                    output.flush()

                    writer.append("\r\n--$boundary--\r\n")
                    writer.flush()

                    val response = conn.inputStream.bufferedReader().readText()
                    Log.i(TAG, "Backend response: $response")

                } catch (e: Exception) {
                    Log.e(TAG, "POST to backend failed", e)
                }
            }.start()
        } else {
            Log.i(TAG, "Manual recording — NOT sending to backend")
        }

    }

    private fun writeSampleToCsv(s: Sample) {
        csvWriter?.append(
            "${s.tsMs}," +
                    "${s.acc[0]},${s.acc[1]},${s.acc[2]}," +
                    "${s.gyro[0]},${s.gyro[1]},${s.gyro[2]}," +
                    "${s.rotv[0]},${s.rotv[1]},${s.rotv[2]}," +
                    "${s.mag[0]},${s.mag[1]},${s.mag[2]}\n"
        )
    }

    // ==========================================
    // Notifications
    // ==========================================

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Motion Sensor Service",
                NotificationManager.IMPORTANCE_HIGH
            )
            val nm = getSystemService(NotificationManager::class.java)
            nm?.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(title: String, content: String): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pending = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentTitle(title)
            .setContentText(content)
            .setStyle(NotificationCompat.BigTextStyle().bigText(content))
            .setContentIntent(pending)
            .setOngoing(true)
            .build()
    }

    private fun sendFinalizeRequest() {
        try {
            val url = java.net.URL(backendUrl("/finalize-session"))
            val conn = url.openConnection() as java.net.HttpURLConnection
            conn.requestMethod = "POST"
            conn.doOutput = true

            val response = conn.inputStream.bufferedReader().readText()
            Log.i(TAG, "FINALIZE response: $response")

        } catch (e: Exception) {
            Log.e(TAG, "Failed finalize-session", e)
        }
    }

    override fun onTaskRemoved(rootIntent: Intent?) {

        // Do NOT restart if user is actively recording
        if (!isRecording) {
            val restart = Intent(applicationContext, SensorService::class.java)
            restart.setPackage(packageName)

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                applicationContext.startForegroundService(restart)
            } else {
                applicationContext.startService(restart)
            }

            Log.w(TAG, "Service restarted after task removed (idle mode)")
        } else {
            Log.w(TAG, "Task removed but recording active — NOT restarting service")
        }

        super.onTaskRemoved(rootIntent)
    }



}
