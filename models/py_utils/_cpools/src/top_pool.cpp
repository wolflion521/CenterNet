#include <torch/torch.h>

#include <vector>

std::vector<at::Tensor> top_pool_forward(
    at::Tensor input
) {
    // Initialize output
    at::Tensor output = at::zeros_like(input);

    /* input should be batchsize,channel,height,width*/

    // Get height
    int64_t height = input.size(2);

    // Copy the last column
    at::Tensor input_temp  = input.select(2, height - 1);
    // Tensor.select
    // select(dim, index) → Tensor
    // Slices the self tensor along the selected dimension at the given index.
    // This function returns a tensor with the given dimension removed
    /* input should be batchsize,channel,height,width
    so, input.select(2,height - 1) should be the last row in Tensor input*/

    at::Tensor output_temp = output.select(2, height - 1);
    /* This line select last row in output tensor*/

    output_temp.copy_(input_temp);
    /* output_temp and input_temp are both extracted rows from output and input tensor*/

    at::Tensor max_temp;
    for (int64_t ind = 1; ind < height; ++ind) {
    /* look at the iterator ind , it is from 1 to height,
     and below the tensor * /
        input_temp  = input.select(2, height - ind - 1);
        /* when ind == 1, this is he height -2 row in input Tensor ,that is to say
        each time ind++ , input_temp goes up in input Tensor
        */
        output_temp = output.select(2, height - ind);
        /****
        when ind == 1, output_temp is the last row in output Tensor,
        so output_temp is one row behind input_temp
        ***/
        max_temp    = output.select(2, height - ind - 1);
        /****
        when ind == 1, height - ind - 1 = height -2
        so this is the line above output_temp
        ***/

        at::max_out(max_temp, input_temp, output_temp);
        /*
        static Tensor &at::max_out(Tensor &out, const Tensor &self, const Tensor &other)
        https://github.com/pytorch/pytorch/blob/88e4cee3e70aac95dd2c18b898808ce3426cb3c9/aten/src/ATen/native/TensorCompare.cpp
        line 179 to check the code of max_out
        didn't understand the original code but I guess meaning is
        maxtemp shape = batchsize, channel, width
        input_temp = batchsize, channel, width
        and for each image in batch:
           for eachchannel in all_channels_of_the_image:
               get the row from input at row 5, name it as input_temp
               get the row from output at row 6, name it as output_temp
               compare two row tensors element by element
               the max_temp only keep the max_element
        */

    }
    /*after the whole for loop , the output tensor would be only the max value in column
    eg:
    input is
    1,    2,    3,    4
    5,   33,    23,   1
    63,   2,    7,    63
    34,   23,    4,   1024

    then the output would be 63,33,23,1024
    */

    return { 
        output
    };
}

std::vector<at::Tensor> top_pool_backward(
    at::Tensor input,
    at::Tensor grad_output
) {
    auto output = at::zeros_like(input);

    int32_t batch   = input.size(0);
    int32_t channel = input.size(1);
    int32_t height  = input.size(2);
    int32_t width   = input.size(3);

    auto max_val = at::zeros(torch::CUDA(at::kFloat), {batch, channel, width});
    auto max_ind = at::zeros(torch::CUDA(at::kLong),  {batch, channel, width});

    auto input_temp = input.select(2, height - 1);
    /* the last row in input Tensor
    shape = batch x channel x width
    */
    max_val.copy_(input_temp);
    /*before this line max_val is totally zeros,
    so Tensor.copy_ is to copy input_temp
    so max_val shape = batch x channel x width
    */

    max_ind.fill_(height - 1);
    /* max_ind shape is {batch, channel, width}
    max_ind is used to store for each col which row contains the largest value in input
    */

    auto output_temp      = output.select(2, height - 1);
    /*
    output is initialized as all zeros,
    right now output_temp is the last row in output
    */

    auto grad_output_temp = grad_output.select(2, height - 1);
    /*grad_output_temp is the last_row in grad_output*/
    output_temp.copy_(grad_output_temp);


    auto un_max_ind = max_ind.unsqueeze(2);
    /*max_ind shape is {batch, channel, width},
    so un_max_ind shape is (batch x channel x 1 x width) */
    auto gt_mask    = at::zeros(torch::CUDA(at::kByte),  {batch, channel, width});
    auto max_temp   = at::zeros(torch::CUDA(at::kFloat), {batch, channel, width});
    for (int32_t ind = 1; ind < height; ++ind) {
    /* look at ind , it is going to iterate from the second last row to the top row of input tensor  */
        input_temp = input.select(2, height - ind - 1);
        /* input_temp will be iterate from the second last row to the top row of input tensor*/
        at::gt_out(gt_mask, input_temp, max_val);
        /* static Tensor &at::gt_out(Tensor &out, const Tensor &self, Scalar other)
        max_val is one row behind input_temp
        so here max_val and input_temp are all from input tensor
        gt = greater than : compare whether the upper row in input is bigger than the lower row
        if greater than output 1 , otherwise output 0

        */

        at::masked_select_out(max_temp, input_temp, gt_mask);
        /*
        max_temp is shape batch,channel,width
        input_temp is the upper row in input Tensor
        gt_mask is whether upper row ele is bigger than lower row
        so only when upper row is bigger , max_temp will update
        eg.input is
        1,    2,    3,    4
        5,   33,    23,   1
        63,   2,    7,    63
        34,   23,    4,   1024

        initialization:
        gt_mask = [0, 0 , 0, 0]
        max_temp = [0, 0 , 0,0]
        max_val = row3
        max_ind = [3,3,3,3]
        output_temp = grad_output last row
        un_max_ind = [[3,3,3,3]]



        ind = 1 ,compare row 2 and row 3
        input_temp = row 2
        gt_mask = [1, 0 , 1, 0]
        max_temp = [63, 0 , 7,0]
        max_val = [63, 23,7,1024]   as the ind goes up max_val will be filled with the largest value in that column
        max_ind = [2 , 3, 2, 3]
        grad_output_temp = grad_output second last row

        ind = 2, compare row 2 and row
        */
        max_val.masked_scatter_(gt_mask, max_temp);
        max_ind.masked_fill_(gt_mask, height - ind - 1);

        grad_output_temp = grad_output.select(2, height - ind - 1).unsqueeze(2);
        /*
        grad_output_temp is loop over grad_output Tensor from the last row to top row
        because of unsqueeze , grad_output_temp should be batch,channel,1,width
        */
        output.scatter_add_(2, un_max_ind, grad_output_temp);
    }

    return {
        output
    };
}

/*  this is how to use cpp extension to define a pytorch function
#include <torch/torch.h>
write a forward function, input a tensor and output a tensor,
write a backward function, input a tensor and grad, output a tensor

*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", &top_pool_forward, "Top Pool Forward",
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "backward", &top_pool_backward, "Top Pool Backward",
        py::call_guard<py::gil_scoped_release>()
    );
}
/*
# first call
# train.py :      nnet = NetworkFactory
# NetworkFactory inherited from kp
# kp.tl_cnvs is the output of make_tl_layer(256)
# tl_pool(256)
# go to pool(256,TopPool,LeftPool)
# go to TopPool
# go to TopPoolFunction
# go to top_pool in cpools/src/setup.py
# go to "src/top_pool.cpp"
*/