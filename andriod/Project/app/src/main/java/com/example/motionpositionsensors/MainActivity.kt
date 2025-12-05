package com.example.motionpositionsensors

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"
    private val REQ_POST_NOTIF = 101

    private var receiverRegistered = false

    // Live UI sensor text fields
    private lateinit var liveAcc: TextView
    private lateinit var liveGyro: TextView
    private lateinit var liveRot: TextView
    private lateinit var liveMag: TextView

    // Start/Stop buttons
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button

    // BroadcastReceiver that handles live sensor updates + saved CSV notification
    private val sensorReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent == null) return

            when (intent.action) {

                SensorService.ACTION_SENSOR_UPDATE -> {
                    val acc = intent.getFloatArrayExtra("acc")
                    val gyro = intent.getFloatArrayExtra("gyro")
                    val rotv = intent.getFloatArrayExtra("rotv")
                    val mag = intent.getFloatArrayExtra("mag")

                    liveAcc.text = if (acc != null)
                        "ACC: X=%.2f Y=%.2f Z=%.2f".format(acc[0], acc[1], acc[2])
                    else "ACC: - - -"

                    liveGyro.text = if (gyro != null)
                        "GYRO: X=%.2f Y=%.2f Z=%.2f".format(gyro[0], gyro[1], gyro[2])
                    else "GYRO: - - -"

                    liveRot.text = if (rotv != null)
                        "ROT: X=%.2f Y=%.2f Z=%.2f".format(rotv[0], rotv[1], rotv[2])
                    else "ROT: - - -"

                    liveMag.text = if (mag != null)
                        "MAG: X=%.2f Y=%.2f Z=%.2f".format(mag[0], mag[1], mag[2])
                    else "MAG: - - -"
                }

                SensorService.ACTION_RECORDING_SAVED -> {
                    val fileName = intent.getStringExtra("fileName")
                    Toast.makeText(
                        this@MainActivity,
                        "Saved: $fileName",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.d(TAG, "onCreate")

        // Bind all TextViews
        liveAcc = findViewById(R.id.live_acc)
        liveGyro = findViewById(R.id.live_gyro)
        liveRot = findViewById(R.id.live_rot)
        liveMag = findViewById(R.id.live_mag)

        // Bind buttons
        btnStart = findViewById(R.id.btn_start)
        btnStop = findViewById(R.id.btn_stop)

        // Start sensor service automatically
        startSensorServiceSafely()

        requestNotificationPermissionIfNeeded()

        // Setup keypad listeners for zones 0–9
        setupZoneClickListeners()

        // Setup Start button manually begins recording without a zone
        btnStart.setOnClickListener {
            val intent = Intent(this, SensorService::class.java).apply {
                action = SensorService.ACTION_START_RECORDING
                putExtra("zoneName", "manual")
            }
            startService(intent)
            Toast.makeText(this, "Manual recording started", Toast.LENGTH_SHORT).show()
        }

        // Setup Stop button
        btnStop.setOnClickListener {
            val intent = Intent(this, SensorService::class.java).apply {
                action = SensorService.ACTION_STOP_RECORDING
            }
            startService(intent)
            Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show()
        }
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.POST_NOTIFICATIONS
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                    REQ_POST_NOTIF
                )
            }
        }
    }

    private fun startSensorServiceSafely() {
        val svcIntent = Intent(this, SensorService::class.java)
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                startForegroundService(svcIntent)
            else
                startService(svcIntent)

            Log.d(TAG, "Requested SensorService start")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start SensorService", e)
        }
    }

    private fun setupZoneClickListeners() {
        val zoneIds = listOf(
            R.id.layout_zone1, R.id.layout_zone2, R.id.layout_zone3,
            R.id.layout_zone4, R.id.layout_zone5, R.id.layout_zone6,
            R.id.layout_zone7, R.id.layout_zone8, R.id.layout_zone9,
            R.id.layout_zone0
        )

        val zoneNumbers = listOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 0)

        zoneIds.zip(zoneNumbers).forEach { (viewId, zoneNumber) ->
            findViewById<LinearLayout>(viewId)?.setOnClickListener {
                startZoneRecording(zoneNumber)
            }
        }
    }

    private fun startZoneRecording(zone: Int) {
        val intent = Intent(this, SensorService::class.java).apply {
            action = SensorService.ACTION_START_RECORDING
            putExtra("zoneName", zone.toString())
        }
        startService(intent)
        Toast.makeText(this, "Recording zone $zone", Toast.LENGTH_SHORT).show()
    }

    override fun onStart() {
        super.onStart()

        val filter = IntentFilter().apply {
            addAction(SensorService.ACTION_SENSOR_UPDATE)
            addAction(SensorService.ACTION_RECORDING_SAVED)
        }

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                registerReceiver(sensorReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
            } else {
                registerReceiver(sensorReceiver, filter)
            }
            receiverRegistered = true
            Log.d(TAG, "sensorReceiver registered")
        } catch (e: Exception) {
            Log.e(TAG, "registerReceiver failed", e)
        }
    }

    override fun onStop() {
        super.onStop()
        if (receiverRegistered) {
            try {
                unregisterReceiver(sensorReceiver)
            } catch (e: Exception) {
                Log.w(TAG, "unregisterReceiver failed", e)
            } finally {
                receiverRegistered = false
            }
        }
    }

    override fun onPause() {
        super.onPause()
        SensorService.appInForeground = false
    }

    override fun onResume() {
        super.onResume()
        SensorService.appInForeground = true
    }

}
