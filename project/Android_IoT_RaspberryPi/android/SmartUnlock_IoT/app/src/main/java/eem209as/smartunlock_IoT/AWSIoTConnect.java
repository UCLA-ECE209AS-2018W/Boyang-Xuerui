package eem209as.smartunlock_IoT;

import android.util.Log;

import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.CognitoCachingCredentialsProvider;
import com.amazonaws.mobileconnectors.iot.AWSIotMqttClientStatusCallback;
import com.amazonaws.mobileconnectors.iot.AWSIotMqttManager;
import com.amazonaws.mobileconnectors.iot.AWSIotMqttNewMessageCallback;
import com.amazonaws.mobileconnectors.iot.AWSIotMqttQos;
import com.amazonaws.regions.Regions;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.UnsupportedEncodingException;

/**
 * Created by boyang on 3/19/18.
 */

public class AWSIoTConnect {
    private static final String LOG_TAG = AWSIoTConnect.class.getCanonicalName();

    // --- Constants to modify per your configuration ---

    // Customer specific IoT endpoint
    // AWS Iot CLI describe-endpoint call returns: XXXXXXXXXX.iot.<region>.amazonaws.com,
    private static final String CUSTOMER_SPECIFIC_ENDPOINT = "a11nf0pk1jaec3.iot.us-east-1.amazonaws.com";
    // Cognito pool ID. For this app, pool needs to be unauthenticated pool with
    // AWS IoT permissions.
    private static final String COGNITO_POOL_ID = "us-east-1:7f87877c-5ed4-4dea-b902-14c2b4950fe3";
    // Region of AWS IoT
    private static final Regions MY_REGION = Regions.US_EAST_1;

    private AWSIotMqttManager mqttManager;

    private AWSCredentials awsCredentials;
    private CognitoCachingCredentialsProvider credentialsProvider;

    //These variable are given or generated by the Activity class
    private static MainActivity mainActivity = null;
    private static String clientId;

    public AWSIoTConnect(MainActivity inputActivity) {
        mainActivity = inputActivity;
    }

    public void onCreateCall() {
        // MQTT client IDs are required to be unique per AWS IoT account.
        // This UUID is "practically unique" but does not _guarantee_
        // uniqueness.
        clientId = mainActivity.clientId;
        // Initialize the AWS Cognito credentials provider
        credentialsProvider = new CognitoCachingCredentialsProvider(
                mainActivity.getApplicationContext(), // context
                COGNITO_POOL_ID, // Identity Pool ID
                MY_REGION // Region
        );

//        Region region = Region.getRegion(MY_REGION);

        // MQTT Client
        mqttManager = new AWSIotMqttManager(clientId, CUSTOMER_SPECIFIC_ENDPOINT);

        // The following block uses IAM user credentials for authentication with AWS IoT.
        //awsCredentials = new BasicAWSCredentials("ACCESS_KEY_CHANGE_ME", "SECRET_KEY_CHANGE_ME");
        //btnConnect.setEnabled(true);


        // The following block uses a Cognito credentials provider for authentication with AWS IoT.

        new Thread(() -> {
            awsCredentials = credentialsProvider.getCredentials();

            mainActivity.runOnUiThread(this::connectToServer);
        }).start();
    }

    private void connectToServer() {
        Log.d(LOG_TAG, "clientId = " + clientId);

        try {
            mqttManager.connect(credentialsProvider, new AWSIotMqttClientStatusCallback() {
                @Override
                public void onStatusChanged(final AWSIotMqttClientStatus status,
                                            final Throwable throwable) {
                    mainActivity.connection = true;
                    Log.d(LOG_TAG, "Status = " + String.valueOf(status));
                }
            });
        } catch (final Exception e) {
            Log.e(LOG_TAG, "Connection error.", e);
        }
    }

    public void sendData(DataClass myData) {
        //first we need to subscribe the listening topic so we can catch the sign up response
        final String listenTopic = "rpiToPhone";
        Log.d(LOG_TAG, "listenTopic = " + listenTopic);


        try {
            mqttManager.subscribeToTopic(listenTopic, AWSIotMqttQos.QOS0, new AWSIotMqttNewMessageCallback() {
                @Override
                public void onMessageArrived(String topic, byte[] data) {
                    try {
                        String message = new String(data, "UTF-8");
                        Log.d(LOG_TAG, "Message arrived:");
                        Log.d(LOG_TAG, "   Topic: " + topic);
                        Log.d(LOG_TAG, " Message: " + message);

                        JSONObject jsonMessage = new JSONObject(message);
                        int result = jsonMessage.getInt("result");
                        myData.result = result;
                        new Thread(mainActivity::dataReceived).start();
                        Log.d(LOG_TAG, "Result: " + result);
//                                   activity.tvLastMessage.setText(String.valueOf(isSignUpSuccess));
//                                    activity.tvLastMessage.setText((String) jsonMessage.get("message"));


                    } catch (UnsupportedEncodingException e) {
                        Log.e(LOG_TAG, "Message encoding error.", e);
                    } catch (JSONException e) {
                        Log.e(LOG_TAG, "JSON creating error.", e);
                    }
                }
            });
        } catch (Exception e) {
            Log.e(LOG_TAG, "Subscription error.", e);
        }


        //then we need to send the sign up request to the server
        final String sendTopic = "phoneToRpi";
        final String msg;
        JSONObject sendContent = new JSONObject();

        Log.d(LOG_TAG, "sendTopic = " + sendTopic);

        try {
            switch(myData.dayStamp){
                case "Monday":
                case "Tuesday":
                case "Wednesday":
                case "Thursday":
                case "Friday":
                    sendContent.put("localDay", "Weekday");
                    break;
                case "Saturday":
                case "Sunday":
                    sendContent.put("localDay", "Weekend");
                    break;
            }
            sendContent.put("localTime", myData.timeStamp.substring(0,2));
            sendContent.put("g", Double.toString(myData.g));
            sendContent.put("latitude", Double.toString(myData.lat));
            sendContent.put("longitude", Double.toString(myData.lng));
            sendContent.put("accuracy", Double.toString(myData.acu));
            sendContent.put("altitude", Double.toString(myData.alt));
            sendContent.put("speed", Double.toString(myData.speed));
            sendContent.put("wifi mac", myData.wifiInfo.get("BSSID"));
            sendContent.put("wifi ssid",  myData.wifiInfo.get("SSID"));
            sendContent.put("wifi signal level", myData.wifiInfo.get("RSSI"));
            sendContent.put("provider", myData.provider);
            sendContent.put("safe", "10");

            msg = sendContent.toString();
            mqttManager.publishString(msg, sendTopic, AWSIotMqttQos.QOS0);
        } catch (JSONException e) {
            Log.e(LOG_TAG, "JSON put error. ", e);
        } catch (IllegalArgumentException e) {
            Log.e(LOG_TAG, "Publish error.", e);
        }
    }

}