package com.example.pytorch_app;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Random;

import com.chaquo.python.*;


public class MainActivity2 extends AppCompatActivity {

    // Elements in the view
    Button btnGenerate;
    ImageView ivImage;
    TextView tvWaiting;

    // Tag used for logging
    private static final String TAG = "MainActivity2";

    // PyTorch model
    Module module;
    Module decoder;

    // Size of the input tensor
    int inSize = 512;

    // Width and height of the output image
    int width = 128;
    int height = 128;

    public void createTempDirectories() {

        //create temp folders
        File tempDir = new File(this.getFilesDir(), "temp");
        if (tempDir.exists()) {
            deleteRecursive(tempDir);
        }
        tempDir.mkdirs();

        File imgDir = new File(tempDir, "img");
        imgDir.mkdirs();

        File motionDir = new File(tempDir, "motion");
        motionDir.mkdirs();

        File attentionDir = new File(tempDir, "attention");
        attentionDir.mkdirs();

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        // Get the elements in the activity
        btnGenerate = findViewById(R.id.btnGenerate);
        ivImage = findViewById(R.id.ivImage);
        tvWaiting = findViewById(R.id.tvWaiting);

        // Load in the model
        try {
            module = LiteModuleLoader.load(assetFilePath("imageGen.pt"));
            decoder = LiteModuleLoader.load(assetFilePath("vgnetmodeloptimised1126.pt"));
        } catch (IOException e) {
            Log.e(TAG, "Unable to load model", e);
        }

        // When the button is clicked, generate a new image
        btnGenerate.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Error handing
                btnGenerate.setClickable(false);
                ivImage.setVisibility(View.INVISIBLE);
                tvWaiting.setVisibility(View.VISIBLE);

                createTempDirectories(); //path : /data/data/com.example.pytorch_app/files

                //load pca and mean
//                try {
//                    // Load 'U_lrw1.npy' from assets and convert to an ND4J array
//                    INDArray uLrw1 = Nd4j.createFromNpyFile(new File("file:///android_asset/U_lrw1.npy"));
//                    // Extract the first 6 columns
//                    INDArray pca1 = uLrw1.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 6));
//                    // Convert ND4J array to PyTorch tensor
//                    float[] pcaData = pca1.data().asFloat();
//                    long[] pcaShape = pca1.shape();
//                    Tensor pca = Tensor.fromBlob(pcaData, pcaShape);
//
//                    // Load 'mean_lrw1.npy' from assets and convert to an ND4J array
//                    INDArray meanLrw1 = Nd4j.createFromNpyFile(new File("file:///android_asset/mean_lrw1.npy"));
//
//                    // Convert ND4J array to PyTorch tensor
//                    float[] meanData = meanLrw1.data().asFloat();
//                    long[] meanShape = meanLrw1.shape();
//                    Tensor mean = Tensor.fromBlob(meanData, meanShape);
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }
//

                Tensor example_image = null,fake_lmark = null,example_landmark = null;

                // Load tensors from the file
                try {
                    example_image = Tensor.fromBlob(loadTensorData("example_image_6.txt"), new long[]{1,3,128,128});
                    fake_lmark = Tensor.fromBlob(loadTensorData("fake_lmark_6.txt"),new long[]{1,149,136});
                    example_landmark = Tensor.fromBlob(loadTensorData("example_landmark_6.txt"),new long[]{1,136});
                    Log.e(TAG, "loaded tensors"+example_image);
                } catch (IOException e) {
                    Log.e(TAG, "Unable to load tensor", e);
                }



                // Prepare the input tensor. This time, its a
                // a single integer value.
                Tensor inputTensor = generateTensor(inSize);
                Log.e(TAG, "inputTensor"+inputTensor);


                // Run the process on a background thread
                Tensor finalExample_image = example_image;
                Tensor finalFake_lmark = fake_lmark;
                Tensor finalExample_landmark = example_landmark;

//                Log.e(TAG,"ok : started");
//                float[] fake_ims, atts ,ms ,extra = decoder.forward(IValue.from(finalExample_image),IValue.from(finalFake_lmark),IValue.from(finalExample_landmark)).toTensor().getDataAsFloatArray();
//                Log.e(TAG,"hurray done");

                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        // Get the output from the model. The
                        // length should be 256*256*3 or 196608
                        // Note that the output is in the layout
                        // [R, G, B, R, G, B, ..., B] and we
                        // have to deal with that.
//                        float[] outputArr  = {0.1F};
//                        Tensor outputArr = module.forward(IValue.from(inputTensor)).toTensor();
//                        Log.e(TAG,"outputArr: "+outputArr);
                        Log.e(TAG,"ok : started");
//                        .toTensor().getDataAsDoubleArray()
                        try{
                            long startTime = System.currentTimeMillis();

                            IValue output = decoder.forward(IValue.from(finalExample_image), IValue.from(finalFake_lmark), IValue.from(finalExample_landmark));
                            long endTime = System.currentTimeMillis();
                            long elapsedTime = endTime - startTime;

                            Log.d("ProcessingSpeed", "Elapsed Time: " + elapsedTime + " milliseconds");
                            IValue[] output_tuple = output.toTuple();
                            Tensor fake_ims = output_tuple[0].toTensor();
                            Log.e(TAG, "Type fake_ims : "+ fake_ims);
//                            Log.e(TAG, "Length fake_ims : "+ fake_ims.length);
                            float[] outputData = fake_ims.getDataAsFloatArray();
                            long[] shape = fake_ims.shape();
//                            Log.e(TAG, "outputData : "+ outputData.length);
//                            Log.e(TAG, "shape : "+ shape[0]);
//                            Log.e(TAG, "shape : "+ shape[1]);
//                            Log.e(TAG, "shape : "+ shape[2]);

                            for (int indx = 0; indx < shape[1]; indx++) {
                                float[] fakeStore = new float[(int) (shape[2] * shape[3] * shape[4])];//[1,149,3,128,128]
                                System.arraycopy(outputData, indx * fakeStore.length, fakeStore, 0, fakeStore.length);
                                long[] reshapeDims = {(int) shape[2], (int) shape[3], (int) shape[4]};
                                Tensor reshapedTensor = Tensor.fromBlob(fakeStore, reshapeDims);

                                int dim0 = 3;
                                int dim1 = 128;
                                int dim2 = 128;
                                float[] newFakeStore = new float[128*128*3];
                                for (int i = 0; i < dim0; i++) {
                                    for (int j = 0; j < dim1; j++) {
                                        for (int k = 0; k < dim2; k++) {
                                            int indexOriginal = i * dim1 * dim2 + j * dim2 + k;
                                            int indexNew = j * dim2 * dim0 + k * dim0 + i;
                                            newFakeStore[indexNew] = fakeStore[indexOriginal];
                                        }
                                    }
                                }
//                                Log.e(TAG, "reshapedTensor : "+reshapedTensor);

//                                float[] reshapedArray = reshapedTensor.getDataAsFloatArray();
                                convertToBitmap(newFakeStore,indx);
                            }

                        }  catch (Error e) {
                            Log.e(TAG, "OMG : ", e);
                        }

//                        float[] fake_ims, atts ,ms ,extra = decoder.forward(IValue.from(finalExample_image),IValue.from(finalFake_lmark),IValue.from(finalExample_landmark)).toTensor().getDataAsFloatArray();
                        Log.e(TAG,"hurray done");








//                        Tensor  fake_lmar = module1.forward(IValue.from(example_landmark, input_mfcc)).toTensor();
                        // Ensure the output array has values between 0 and 255
//                        for (int i = 0; i < outputArr.length; i++) {
//                            outputArr[i] = Math.min(Math.max(outputArr[i], 0), 255);
//                        }

                        // Create a RGB bitmap of the correct shape
//                        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
//
//                        // Iterate over all values in the output tensor
//                        // and put them into the bitmap
//                        int loc = 0;
//                        for (int y = 0; y < width; y++) {
//                            for (int x = 0; x < height; x++) {
//                                bmp.setPixel(x, y, Color.rgb((int)outputArr[loc], (int)outputArr[loc+1], (int)outputArr[loc+2]));
//                                loc += 3;
//                            }
//                        }

                        // The output of the network is no longer needed
//                        outputArr = null;
//
//                        // Resize the bitmap to a larger image
//                        bmp = Bitmap.createScaledBitmap(
//                                bmp, 512, 512, false);
//
//                        // Display the image
//                        Bitmap finalBmp = bmp;
//                        runOnUiThread(new Runnable() {
//                            @Override
//                            public void run() {
//                                ivImage.setImageBitmap(finalBmp);
//
//                                // Error handing
//                                btnGenerate.setClickable(true);
//                                tvWaiting.setVisibility(View.INVISIBLE);
//                                ivImage.setVisibility(View.VISIBLE);
//                            }
//                        }
//                        );

                    }
                }).start();

            }
        });
    }

    private void convertToBitmap(float[] outputArr,int imageIndex) {

//         Ensure the output array has values between 0 and 255
                        for (int i = 0; i < outputArr.length; i++) {

                            outputArr[i] = Math.min(Math.max(outputArr[i], 0), 1)*255;
                            Log.e(TAG, "outputArr : "+outputArr[i]);
                        }

//         Create a RGB bitmap of the correct shape
                        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);

                        // Iterate over all values in the output tensor
                        // and put them into the bitmap
                        int loc = 0;
                        for (int y = 0; y < width; y++) {
                            for (int x = 0; x < height; x++) {
                                bmp.setPixel(x, y, Color.rgb((int)outputArr[loc], (int)outputArr[loc+1], (int)outputArr[loc+2]));
                                loc += 3;
                            }
                        }

//         The output of the network is no longer needed
                        outputArr = null;

                        // Resize the bitmap to a larger image
                        bmp = Bitmap.createScaledBitmap(
                                bmp, 512, 512, false);

                        // Display the image
                        Bitmap finalBmp = bmp;

                        String filePath = "/data/data/com.example.pytorch_app/files/temp/img/" + imageIndex + ".png";
                        File file = new File(filePath);
                        FileOutputStream outStream;
                        try {
                            outStream = new FileOutputStream(file);
                            finalBmp.compress(Bitmap.CompressFormat.PNG, 100, outStream);
                            outStream.flush();
                            outStream.close();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }



                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                ivImage.setImageBitmap(finalBmp);

                                // Error handing
                                btnGenerate.setClickable(true);
                                tvWaiting.setVisibility(View.INVISIBLE);
                                ivImage.setVisibility(View.VISIBLE);
                            }
                        }
                        );
    }

    // Generate a tensor of random doubles given the size of
    // the tensor to generate
    public Tensor generateTensor(int size) {
        // Create a random array of doubles
        Random rand = new Random();
        double[] arr = new double[size];
        for (int i = 0; i < size; i++) {
            arr[i] = rand.nextGaussian();
        }

        // Create the tensor and return it
        long[] s = {1, size};
        return Tensor.fromBlob(arr, s);
    }


//     Given the name of the pytorch model, get the path for that model
    public String assetFilePath(String assetName) throws IOException {
        File file = new File(this.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = this.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }


    public void deleteRecursive(File fileOrDirectory) {
        if (fileOrDirectory.isDirectory()) {
            for (File child : fileOrDirectory.listFiles()) {
                deleteRecursive(child);
            }
        }
        fileOrDirectory.delete();
    }
    // Utility function to load tensor data from file
    private float[] loadTensorData(String assetFilePath) throws IOException {

        InputStream is = getAssets().open(assetFilePath);

        InputStreamReader inputStreamReader = new InputStreamReader(is);
        BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
        String line = bufferedReader.readLine();
        String[] values = line.split(","); // Assuming values are separated by commas

//        Log.e(TAG,"values : " + values);
        float[] floatArray = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            floatArray[i] = Float.parseFloat(values[i]);
        }
        Log.e(TAG,"floatArray : " + floatArray);

        return floatArray;

//        ByteArrayOutputStream bos = new ByteArrayOutputStream();
//        byte[] buffer = new byte[1024];
//        int bytesRead;
//        while ((bytesRead = is.read(buffer)) != -1) {
//            bos.write(buffer, 0, bytesRead);
//        }
//        Log.e(TAG,"bos.shape: "+bos.toString().length());
//        return bos.toByteArray();
    }


}