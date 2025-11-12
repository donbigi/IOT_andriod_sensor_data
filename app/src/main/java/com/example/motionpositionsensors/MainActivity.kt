package com.example.motionpositionsensors

import android.Manifest
import android.app.Dialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.Window
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private val REQ_POST_NOTIF = 101

    // zone dialog UI
    private var zoneDialog: Dialog? = null
    private var tvAcc: TextView? = null
    private var tvGyro: TextView? = null
    private var tvRot: TextView? = null
    private var tvMag: TextView? = null
    private var btnClose: Button? = null

    private var receiverRegistered = false

    private val sensorReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            Log.v(TAG, "sensorReceiver.onReceive: action=${intent?.action}")
            if (intent == null) return
            when (intent.action) {
                SensorService.ACTION_SENSOR_UPDATE -> {
                    val acc = intent.getFloatArrayExtra("acc")
                    val gyro = intent.getFloatArrayExtra("gyro")
                    val rotv = intent.getFloatArrayExtra("rotv")
                    val mag = intent.getFloatArrayExtra("mag")
                    Log.v(TAG, "Received sensor update: acc=${acc?.contentToString()}, gyro=${gyro?.contentToString()}")

                    // Update dialog if open — otherwise ignore
                    if (acc != null && tvAcc != null) {
                        tvAcc?.text = "ACC: X=${"%.2f".format(acc[0])}, Y=${"%.2f".format(acc[1])}, Z=${"%.2f".format(acc[2])}"
                    } else {
                        tvAcc?.text = "ACC: X=-, Y=-, Z=-"
                    }

                    if (gyro != null && tvGyro != null) {
                        tvGyro?.text = "GYRO: X=${"%.2f".format(gyro[0])}, Y=${"%.2f".format(gyro[1])}, Z=${"%.2f".format(gyro[2])}"
                    } else {
                        tvGyro?.text = "GYRO: X=-, Y=-, Z=-"
                    }

                    if (rotv != null && tvRot != null) {
                        tvRot?.text = "ROT: X=${"%.2f".format(rotv[0])}, Y=${"%.2f".format(rotv[1])}, Z=${"%.2f".format(rotv[2])}"
                    } else {
                        tvRot?.text = "ROT: X=-, Y=-, Z=-"
                    }

                    if (mag != null && tvMag != null) {
                        tvMag?.text = "MAG: X=${"%.2f".format(mag[0])}, Y=${"%.2f".format(mag[1])}, Z=${"%.2f".format(mag[2])}"
                    } else {
                        tvMag?.text = "MAG: X=-, Y=-, Z=-"
                    }
                }
                SensorService.ACTION_RECORDING_SAVED -> {
                    val fileName = intent.getStringExtra("fileName")
                    Log.i(TAG, "Received recording saved broadcast: $fileName")
                    Toast.makeText(this@MainActivity, "Recording saved: $fileName", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "onCreate")
        setContentView(R.layout.activity_main)

        // Start the sensor service immediately when app launches
        startSensorServiceSafely()

        // Request notification permission (Android 13+) so updates can be shown
        requestNotificationPermissionIfNeeded()

        // Setup click listeners for zone dialogs
        setupZoneClickListeners()
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
        val svc = Intent(this, SensorService::class.java)
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(svc) else startService(svc)
            Log.d(TAG, "Requested start of SensorService")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start SensorService", e)
        }
    }

    private fun setupZoneClickListeners() {
        val zoneIds = listOf(
            R.id.layout_zone1, R.id.layout_zone2, R.id.layout_zone3,
            R.id.layout_zone4, R.id.layout_zone5, R.id.layout_zone6,
            R.id.layout_zone7, R.id.layout_zone8, R.id.layout_zone9
        )

        zoneIds.forEachIndexed { index, id ->
            findViewById<LinearLayout>(id)?.setOnClickListener {
                showZoneDataDialog(index + 1)
            }
        }
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
            Log.d(TAG, "Receiver registered")
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

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_POST_NOTIF && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startSensorServiceSafely()
        }
    }

    private fun showZoneDataDialog(zoneNumber: Int) {
        zoneDialog = Dialog(this)
        zoneDialog?.requestWindowFeature(Window.FEATURE_NO_TITLE)
        zoneDialog?.setContentView(R.layout.dialog_zone_data)
        zoneDialog?.setCancelable(false)

        tvAcc = zoneDialog?.findViewById(R.id.text_acc_data)
        tvGyro = zoneDialog?.findViewById(R.id.text_gyro_data)
        tvRot = zoneDialog?.findViewById(R.id.text_rot_data)
        tvMag = zoneDialog?.findViewById(R.id.text_mag_data)
        btnClose = zoneDialog?.findViewById(R.id.button_close)

        // set placeholders right away
        tvAcc?.text = "ACC: X=-, Y=-, Z=-"
        tvGyro?.text = "GYRO: X=-, Y=-, Z=-"
        tvRot?.text = "ROT: X=-, Y=-, Z=-"
        tvMag?.text = "MAG: X=-, Y=-, Z=-"

        val zoneName = "ZONE$zoneNumber"

        val startIntent = Intent(this, SensorService::class.java).apply {
            action = SensorService.ACTION_START_RECORDING
            putExtra("zoneName", zoneName)
        }
        startService(startIntent)

        btnClose?.setOnClickListener {
            val stopIntent = Intent(this, SensorService::class.java).apply {
                action = SensorService.ACTION_STOP_RECORDING
            }
            startService(stopIntent)
            zoneDialog?.dismiss()
        }

        zoneDialog?.show()
    }
}
