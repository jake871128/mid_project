
#include "mbed.h"
#include <cmath>
#include "DA7212.h"
#include "uLCD_4DGL.h"

#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define bufferLength (32)

#define signalLength (294)

#define n (3)

DA7212 audio;

int16_t waveform[kAudioTxBufferSize];

EventQueue queue(32 * EVENTS_EVENT_SIZE);

EventQueue queue_menu(32 * EVENTS_EVENT_SIZE);

Thread t;

Thread t1; 

Thread t2(osPriorityNormal, 120 * 1024 /*120K stack size*/);

Serial pc(USBTX, USBRX);

InterruptIn button_current_function(SW2);

InterruptIn button_mode_selection(SW3);

uLCD_4DGL uLCD(D1, D0, D2);

bool stop = 1;

float volume;

int num_song = 0;

void loadSignal();
void loadSignalHandler(void) {queue.call(loadSignal);}
int idC = 0;
int send[signalLength];
int sending=1;
char serialInBuffer[bufferLength];
DigitalOut green_led(LED2);
int serialCount = 0;
int song[n][49]={};
int noteLength[n][49]={}; 
int button_count=0;
int change=-2;
int choose = 0;
int gesture_index=0;

void show_info();
void loadSignal(void)
{

  int a,b;
  int i = 0;
  sending=1;
  serialCount = 0;
  stop=1;
  green_led = 0;
  while(i < signalLength)
  {
    
    if(pc.readable())

    {

      serialInBuffer[serialCount] = pc.getc();

      serialCount++;

      if(serialCount == 3)
      {
        a=i/49;
        b=i%49;
      //  serialInBuffer[serialCount] = '\0';
        send[i] = (int) atoi(serialInBuffer);
        if(a<n){
        song[a][b]=send[i];
           pc.printf("a=%d, b=%d, song=%d\r\n",a,b,song[a][b]);
        }else{
          noteLength[a-n][b]=send[i];
           pc.printf("a=%d, b=%d, noteLength=%d\r\n",a-n,b,noteLength[a-n][b]);
        }
        
        serialCount = 0;

        i++;

      }

    }

  }
  green_led = 1;
  sending=0;
  stop=1;

  num_song = 0;
}


// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}



void current_function(){
    
    //pause || playing
    if(choose == 0)     stop = !stop;

    else if(choose == 1)     change = -1;

  
}




void mode_selection(){
    //0 -> 1
    if (change ==0 )  change = 1 ;

    
    else if(change == 1){
      //1 -> 0
      if(choose == 0){
          num_song = num_song + 1;   
          if(num_song == 3){
              num_song = 0;
          } 
          change =0;
      }
      //1 -> 0
      else if (choose == 1 ){
          num_song = num_song - 1;   
          if(num_song == -1){
              num_song = 2;
          } 
          change = 0;
      }
      //1 -> 2
      else if(choose == 2 ){
          change = 2;
      }
    } 
    //2 -> 0
    else if(change == 2 ){
      if(choose == 0)        num_song = 0;
      else if(choose == 1)   num_song = 1;
      else if(choose == 2)   num_song = 2;
      change = 0;
    }
}

void DNN(){
      // Create an area of memory to use for input, output, and intermediate arrays.
    // The size of this will depend on the model you're using, and may need to be
    // determined by experimentation.
    constexpr int kTensorArenaSize = 60 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];

    // Whether we should clear the buffer next time we fetch data
    bool should_clear_buffer = false;
    bool got_data = false;

    // The gesture index of the prediction
    

    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      error_reporter->Report(
          "Model provided is schema version %d not equal "
          "to supported version %d.",
          model->version(), TFLITE_SCHEMA_VERSION);
          return -1;
    }
    
      // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // An easier approach is to just use the AllOpsResolver, but this will
    // incur some penalty in code space for op implementations that are not
    // needed by this graph.
    static tflite::MicroOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
        tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                                tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                                tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                tflite::ops::micro::Register_SOFTMAX());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                              tflite::ops::micro::Register_RESHAPE(), 1);
    // Build an interpreter to run the model with
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    tflite::MicroInterpreter* interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors
    interpreter->AllocateTensors();

    // Obtain pointer to the model's input tensor
    TfLiteTensor* model_input = interpreter->input(0);
    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != config.seq_length) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32)) {
      error_reporter->Report("Bad input tensor parameters in model");
      return -1;
    }

    int input_length = model_input->bytes / sizeof(float);

    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
    if (setup_status != kTfLiteOk) {
      error_reporter->Report("Set up failed\n");
      return -1;
    }

    error_reporter->Report("Set up successful...\n");

    while (true) {

        // Attempt to read new data from the accelerometer
        got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                    input_length, should_clear_buffer);

        // If there was no new data,
        // don't try to clear the buffer again and wait until next time
        if (!got_data) {
          should_clear_buffer = false;
          continue;
        }

        // Run inference, and report any error
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          error_reporter->Report("Invoke failed on index: %d\n", begin_index);
          continue;
        }

        // Analyze the results to obtain a prediction
        gesture_index = PredictGesture(interpreter->output(0)->data.f);

        // Clear the buffer next time we read data
        should_clear_buffer = gesture_index < label_num;

        // Produce an output
        if (gesture_index < label_num) {
          error_reporter->Report(config.output_message[gesture_index]);
          pc.printf("gesture_index == %d",gesture_index);
          if(gesture_index == 0 ){
            choose = choose - 1;
            if(choose == -1)  choose = 2;
          }
          else if (gesture_index == 1){
              choose = choose - 1;
              if(choose == -1)  choose = 2;
          }
          else if(gesture_index == 2){
              choose = choose + 1;
              if(choose == 3)   choose = 0;
          }
        }
        
        
        wait(0.5);
    }
}





void playNote(int freq){

  for (int i = 0; i < kAudioTxBufferSize; i++)

  {

    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1) * volume);

  }

  // the loop below will play the note for the duration of 1s

  for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)

  {

    audio.spk.play(waveform, kAudioTxBufferSize);
    // idC = audio.spk.play(waveform, kAudioTxBufferSize);

  }

}

void show_info(){

  while(1){
    //List one
    if (change == 0){
        uLCD.color(BLUE);

        //twinkle
        if(num_song == 0){
          uLCD.reset();
          uLCD.cls();
          uLCD.locate(0,1);
          uLCD.printf("Twinkle Twinkle \nLittle Star\n");  
        }

        else if(num_song == 1){
            uLCD.reset();
            uLCD.cls();
            uLCD.locate(0,1);
            uLCD.printf("Lightly Row\n");  

        }
        else if(num_song == 2){
            uLCD.reset();
            uLCD.cls();
            uLCD.locate(0,1);
            uLCD.printf("Two Tigers\n"); 

        }
        if(stop == 0)       uLCD.printf("(playing)\n\n");
        else if(stop == 1)  uLCD.printf("(Pause)\n\n");  


        if(choose == 0) {
            uLCD.color(RED);
            if(stop == 0 )        uLCD.printf("Pause\n");
            else if (stop == 1)   uLCD.printf("Playing\n");
            uLCD.color(WHITE);
            uLCD.printf("Interrupt\n");
            uLCD.printf("Load Song\n");   
        }
        else if(choose == 1) {
            uLCD.color(WHITE);
            if(stop == 0 )        uLCD.printf("Pause\n");
            else if (stop == 1)   uLCD.printf("Playing\n");
            uLCD.color(RED);
            uLCD.printf("Interrupt\n");
            uLCD.color(WHITE);
            uLCD.printf("Load Song\n");   
        }
        else if(choose == 2) {
            uLCD.color(WHITE);
            if(stop == 0 )        uLCD.printf("Pause\n");
            else if (stop == 1)   uLCD.printf("Playing\n");
            uLCD.printf("Interrupt\n");
            uLCD.color(RED);
            uLCD.printf("Load Song\n");   
        }
        
     
        
    }

    //List two
    else if (change  == 1) {

        //choose forward
        if(choose == 0 ){  
            uLCD.reset();
            uLCD.cls();
            uLCD.locate(0,1);
            uLCD.color(RED);
            uLCD.printf("Forward\n\n");
            uLCD.color(WHITE);
            uLCD.printf("Backward\n\n");
            uLCD.printf("Change Songs\n\n");
        }
        //choose backward
        else if(choose ==1 ){
            uLCD.reset();
            uLCD.cls();
            uLCD.locate(0,1);
            uLCD.color(WHITE);
            uLCD.printf("Forward\n\n");
            uLCD.color(RED);
            uLCD.printf("Backward\n\n");
            uLCD.color(WHITE);
            uLCD.printf("Change Songs\n\n");
        }
        //choose change songs
        else {
            uLCD.reset();
            uLCD.cls();
            uLCD.locate(0,1);
            uLCD.color(WHITE);
            uLCD.printf("Forward\n\n");
            uLCD.printf("Backward\n\n");
            uLCD.color(RED);
            uLCD.printf("Change Songs\n\n");
        }
    }
    //List Three
    else if (change == 2){

        //choose twinkle
        if(choose == 0 ){  
            uLCD.reset();
            uLCD.cls();
            uLCD.locate(0,1);
            uLCD.color(RED);
            uLCD.printf("Twinkle Twinkle \nLittle Star\n\n");
            uLCD.color(WHITE);
            uLCD.printf("Lightly Row\n\n");
            uLCD.printf("Two Tigers\n\n");
        }
        //choose lightly
        else if(choose ==1 ){
            uLCD.reset();
            uLCD.cls();
            uLCD.locate(0,1);
            uLCD.color(WHITE);
            uLCD.printf("Twinkle Twinkle \nLittle Star\n\n");
            uLCD.color(RED);
            uLCD.printf("Lightly Row\n\n");
            uLCD.color(WHITE);
            uLCD.printf("Two Tigers\n\n");
        }
        //choose change songs
        else {
            uLCD.reset();
            uLCD.cls();
            uLCD.locate(0,1);
            uLCD.color(WHITE);
            uLCD.printf("Twinkle Twinkle \nLittle Star\n\n");
            uLCD.printf("Lightly Row\n\n");
            uLCD.color(RED);
            uLCD.printf("Two Tigers\n\n");
        }
    }

    else if (change == -1 ){
        uLCD.reset();
        uLCD.cls();
        uLCD.locate(0,1);
        uLCD.color(WHITE);
        uLCD.printf("Shut Down \nBye La!\n");
    }

    else if (change == -2 ){
        uLCD.reset();
        uLCD.cls();
        uLCD.locate(0,1);
        uLCD.color(WHITE);
        uLCD.printf("Loading...\n");
        
    }
  
  }

}

void play(){
int length;
    for(int i = 0; i < 49; i++){
      
        pc.printf("num_song == %d",num_song);
        pc.printf("change == %d",change);
        pc.printf("Send Finished!\r\n");

        length = noteLength[num_song][i];

        while(length--){

              if(stop == 0){
                  volume = 1;
                  queue.call(playNote, song[num_song][i]);
              }
              else{
                  volume = 0;
                  // queue.event(stopPlayNote);
                  queue.call(playNote, song[num_song][i]);
                  // length++;
                  if(i>0) i--;
                  else i = 0 ;  
              }

              if(length <= 1) wait(1.0);

        }

    }
    
}


int main(void){
  
    Thread tt;
    tt.start(queue.event(loadSignalHandler));
    t.start(callback(&queue, &EventQueue::dispatch_forever));
    t1.start(show_info);  
    button_current_function.rise(&current_function);

    button_mode_selection.rise(&mode_selection);
    wait(60);
    change=0;
    t2.start(DNN);
      
    play();


  

}
