<?php
namespace App\Services;

use Illuminate\Support\Facades\Log;

class FANNService
{
    public function initial()
    {
        $num_input = 12;
        $num_output = 12;
        $num_layers = 5;
        $num_neurons_hidden = 100;
        $desired_error = 0.01;
        $max_epochs = 5000;
        $epochs_between_reports = 10;

        $ann = fann_create_standard(
            $num_layers,
            $num_input,
            $num_neurons_hidden,
            $num_neurons_hidden,
            $num_neurons_hidden,
            $num_output
        );

        if ($ann) {
            fann_set_activation_function_hidden($ann, FANN_SIGMOID_SYMMETRIC);
            fann_set_activation_function_output($ann, FANN_SIGMOID_SYMMETRIC);

            $filename = dirname(__FILE__) . '/xor.data';
            if (fann_train_on_file($ann, $filename, $max_epochs, $epochs_between_reports, $desired_error))
                fann_save($ann, dirname(__FILE__) . "/xor_float.net");

            fann_destroy($ann);
        }
    }

    public function train()
    {

        $train_file = (dirname(__FILE__) . "/xor_float.net");
        if (!is_file($train_file))
            die("The file xor_float_1.net has not been created! Please run simple_train.php to generate it");

        $origin_ann = fann_create_from_file($train_file);

        $num_input = 12;
        $num_output = 12;
        $num_layers = 5;
        $num_neurons_hidden = 100;
        $desired_error = 0;
        $max_epochs = 500;
        $epochs_between_reports = 10;

        $ann = fann_create_standard(
            $num_layers,
            $num_input,
            $num_neurons_hidden,
            $num_neurons_hidden,
            $num_neurons_hidden,
            $num_output
        );

        if (!$ann)
            die("ANN could not be created");


        fann_set_activation_function_hidden($ann, FANN_SIGMOID_SYMMETRIC);
        fann_set_activation_function_output($ann, FANN_SIGMOID_SYMMETRIC);

        $filename = dirname(__FILE__) . '/xor.data';

        $file_resource = fann_read_train_from_file($filename);

        $origin_MSE = fann_test_data($origin_ann , $file_resource);

        if (fann_train_on_file($ann, $filename, $max_epochs, $epochs_between_reports, $desired_error)) {
            $new_MSE = fann_test_data($ann , $file_resource);
            if($new_MSE < $origin_MSE){
                echo "fann_save()\n$new_MSE:$origin_MSE\n";
                fann_save($ann, dirname(__FILE__) . "/xor_float.net");
                Log::info("fann_save()\n$new_MSE:$origin_MSE\n");
            } else {
                echo "fann fail to save\n$new_MSE:$origin_MSE\n";
                Log::info("fann fail to save\n$new_MSE:$origin_MSE\n");
            }
        } else {
            echo 'fann_train_on_file() fail';
        }
        fann_destroy($ann);
    }

    public function test()
    {
        $train_file = (dirname(__FILE__) . "/xor_float.net");
        if (!is_file($train_file))
            die("The file xor_float.net has not been created!");

        $ann = fann_create_from_file($train_file);
        if (!$ann)
            die("ANN could not be created");

        $file_resource = fann_read_train_from_file((dirname(__FILE__) . '/xor.data'));
        if (!is_resource($file_resource))
            die("resource fail");

        $input = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1];
        $calc_out = fann_run($ann, $input);
        printf(
            "test (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f) \n"
             . "->   (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f)\n",
            $input[0], $input[1], $input[2], $input[3], $input[4], $input[5],
            $input[6], $input[7], $input[8], $input[9], $input[10], $input[11],
            $calc_out[0], $calc_out[1], $calc_out[2], $calc_out[3], $calc_out[4], $calc_out[5],
            $calc_out[6], $calc_out[7], $calc_out[8], $calc_out[9], $calc_out[10], $calc_out[11]
        );

        echo fann_test_data($ann , $file_resource) . "\n";
        fann_destroy($ann);
    }
}