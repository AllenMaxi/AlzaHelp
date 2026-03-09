package com.alzahelp.mobile;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;
import com.alzahelp.mobile.location.AlzaBackgroundLocationPlugin;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(AlzaBackgroundLocationPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
