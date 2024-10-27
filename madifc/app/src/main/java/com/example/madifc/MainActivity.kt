package com.example.madifc

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var captureImageLauncher: ActivityResultLauncher<Intent>
    private lateinit var importImageLauncher: ActivityResultLauncher<Intent>
    private lateinit var classIndices: Map<Int, String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)
        val captureButton: Button = findViewById(R.id.openbutton)
        val galleryButton: Button = findViewById(R.id.gallery)

        requestPermissions()

        try {
            interpreter = loadModelFile()
            classIndices = loadClassIndices()
        } catch (e: IOException) {
            e.printStackTrace()
            Toast.makeText(this, "Model file not found!", Toast.LENGTH_SHORT).show()
            return
        }

        captureImageLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val photo: Bitmap? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    result.data?.getParcelableExtra("data", Bitmap::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    result.data?.getParcelableExtra("data")
                }

                photo?.let {
                    imageView.setImageBitmap(it)
                    runModel(it)
                } ?: run {
                    Toast.makeText(this, getString(R.string.capture_failed), Toast.LENGTH_SHORT).show()
                }
            }
        }

        importImageLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val uri = result.data?.data
                uri?.let { it ->
                    val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(it))
                    bitmap?.let {
                        imageView.setImageBitmap(it)
                        runModel(it)
                    } ?: run {
                        Toast.makeText(this, getString(R.string.error_importing_image), Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }

        captureButton.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            captureImageLauncher.launch(intent)
        }

        galleryButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            importImageLauncher.launch(intent)
        }
    }

    private fun requestPermissions() {
        val permissions = mutableListOf(Manifest.permission.CAMERA)

        // Add the appropriate permission based on Android version
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }

        // Handle Selected Photos Access on Android 14+
        if (Build.VERSION.SDK_INT >= 34) {
            permissions.add(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)
        }

        if (permissions.any { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }) {
            ActivityCompat.requestPermissions(this, permissions.toTypedArray(), 100)
        }
    }

    private fun loadModelFile(): Interpreter {
        val assetFileDescriptor = assets.openFd("madifc.tflite")
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val mappedByteBuffer: MappedByteBuffer =
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        return Interpreter(mappedByteBuffer)
    }

    private fun runModel(bitmap: Bitmap) {
        // Preprocess the image to match model input dimensions and normalize values
        val inputBitmap = Bitmap.createScaledBitmap(bitmap, 299, 299, true)
        val input = ByteBuffer.allocateDirect(299 * 299 * 3 * 4)  // Float (4 bytes) for each pixel in RGB
        input.rewind()

        for (y in 0 until 299) {
            for (x in 0 until 299) {
                val pixel = inputBitmap.getPixel(x, y)
                input.putFloat((pixel shr 16 and 0xFF) / 255.0f)  // Red
                input.putFloat((pixel shr 8 and 0xFF) / 255.0f)   // Green
                input.putFloat((pixel and 0xFF) / 255.0f)        // Blue
            }
        }

        val outputArray = Array(1) { FloatArray(classIndices.size) }
        interpreter.run(input, outputArray)

        // Find the predicted label with the highest confidence
        val maxIndex = outputArray[0].indices.maxByOrNull { outputArray[0][it] } ?: -1
        val label = getLabelForIndex(maxIndex)
        resultTextView.text = getString(R.string.prediction_text, label)
    }

    private fun loadClassIndices(): Map<Int, String> {
        val json = assets.open("class_indices.json").bufferedReader().use { it.readText() }
        val type = object : TypeToken<Map<String, Int>>() {}.type
        val rawIndices: Map<String, Int> = Gson().fromJson(json, type)
        return rawIndices.entries.associateBy({ it.value }, { it.key })
    }

    private fun getLabelForIndex(index: Int): String {
        return classIndices[index] ?: "Unknown"
    }

    override fun onDestroy() {
        interpreter.close()
        super.onDestroy()
    }
}
