<?php
namespace App\Services;

class FANNService
{
    public function train()
    {
        $num_input = 3;
        $num_output = 1;
        $num_layers = 3;
        $num_neurons_hidden = 10;
        $desired_error = 0.00000000000000000000001;
        $max_epochs = 500000;
        $epochs_between_reports = 10;

        $ann = fann_create_standard($num_layers, $num_input, $num_neurons_hidden, $num_output);

        if ($ann) {
            fann_set_activation_function_hidden($ann, FANN_SIGMOID_SYMMETRIC);
            fann_set_activation_function_output($ann, FANN_SIGMOID_SYMMETRIC);

            $filename = dirname(__FILE__) . '/xor.data';
            if (fann_train_on_file($ann, $filename, $max_epochs, $epochs_between_reports, $desired_error))
                fann_save($ann, dirname(__FILE__) . "/xor_float.net");

            fann_destroy($ann);
        }
    }

    public function test()
    {
        $train_file = (dirname(__FILE__) . "/xor_float.net");
        if (!is_file($train_file))
            die("The file xor_float.net has not been created! Please run simple_train.php to generate it");

        $ann = fann_create_from_file($train_file);
        if (!$ann)
            die("ANN could not be created");

        $input = array(1, 0, 0);
        $calc_out = fann_run($ann, $input);
        printf("xor test (%f,%f, %f) -> %f\n", $input[0], $input[1], $input[2], $calc_out[0]);
        fann_destroy($ann);
    }
}