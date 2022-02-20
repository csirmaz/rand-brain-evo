#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

// Commands are used to construct a network and form the gene sequences / threads
#define CMD_NEW_WEIGHT 1
#define CMD_NEW_SUMSI 2
#define CMD_SUMSI_TO_WEIGHT_IN 3
#define CMD_SUMSI_TO_WEIGHT_CTRL 4
#define CMD_WEIGHT_TO_SUMSI_IN 5
#define CMD_WEIGHT_TO_WEIGHT_CTRL 6
#define CMD_POP_WEIGHT 7
#define CMD_POP_SUMSI 8
#define CMD_WEIGHT_TO_INPUT 9
#define CMD_SUMSI_TO_OUT 10
#define CMD_CALL_THREAD 11 // TODO Not implemented yet

#define NUM_INPUTS 7 // red_example(x,y), blue_example(x,y), question(x,y), energy
#define MAX_WEIGHTS 10000
#define MAX_SUMSIS 100

#define W_PIN__NUM 6
#define W_PIN_OUT 0
#define W_PIN_OUT_TYPE 1
#define W_PIN_IN 2
#define W_PIN_IN_TYPE 3
#define W_PIN_CTRL 4
#define W_PIN_CTRL_TYPE 5

#define TYPE_WEIGHT_CTRL 1
#define TYPE_WEIGHT_OUT 2
#define TYPE_SUMSI_IN 3
#define TYPE_SUMSI_OUT 4
#define TYPE_GLOBAL_IN 5

struct brain_t {
    // Weight units have two inputs (input, control) and one output
    // We have a stack of weights for construction, but really they are always numbered sequentially, so we only need a number
    int weight_stack;

    // Sumsi units (sum and sigmoid) can have many inputs and one output
    // We have a stack of them for construction, but really they are always numbered sequentially, so we only need a number
    int sumsi_stack;
    
    // Outgoing connections from a weight unit
    // weight_conn[i][W_PIN_OUT] -- which whatever the output is going to
    // weight_conn[i][W_PIN_OUT_TYPE] -- whether the output is connected to a TYPE_WEIGHT_CTRL or a TYPE_SUMSI_IN
    // weight_conn[i][W_PIN_IN] -- which whatever the input is coming from
    // weight_conn[i][W_PIN_IN_TYPE] -- whether it is TYPE_GLOBAL_IN or TYPE_SUMSI_OUT
    // weight_conn[i][W_PIN_CTRL] -- which whatever the control is coming from
    // weight_conn[i][W_PIN_CTRL_TYPE] -- whether it is TYPE_WEIGHT_OUT or TYPE_SUMSI_OUT
    int weight_conn[MAX_WEIGHTS][W_PIN__NUM]
    
    // Which weights are inputs connected to? (For sanity checking)
    int input_conn[NUM_INPUTS];
    
    // Which sumsi does the output come from?
    int output;
}


void die(char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

// Initialise a brain for construction
void brain_constr_init(const struct brain_t *brain) {
    int i, j;
    brain->weight_stack = 1; // start with 1 so 0 can mean unconnected
    brain->sumsi_stack = 1; // start with 1 so 0 can mean unconnected
    for(i=0; i<NUM_INPUTS; i++) { brain->input_conn[i] = 0; } // unconnected
    brain->output = 0;
    for(i=0; i<MAX_WEIGHTS; i++) for(j=0; j<W_PIN__NUM; j++) brain->weight_conn[i][j] = 0;
}

// Construction: process a command in the gene sequence
void brain_constr_process_command(const struct brain_t *brain, int command, int ix) {
    int p;

    switch(command) {
        case CMD_NEW_WEIGHT: // create new weight unit and push
            brain->weight_stack++;
            if(brain->weight_stack >= MAX_WEIGHTS) { die("Too many weights"); }
            break;
        case CMD_NEW_SUMSI: // create new sumsi unit and push
            brain->sumsi_stack++;
            if(brain->sumsi_stack >= MAX_SUMSIS) { die("Too many sumsis"); }
            break;
        case CMD_SUMSI_TO_WEIGHT_IN: // connect latest sumsi.out to weight[-ix].in
            // TODO Allow override?
            p = brain->weight_stack - ix;
            if(p >= 1) {
                brain->weight_conn[p][W_PIN_IN_TYPE] = TYPE_SUMSI_OUT;
                brain->weight_conn[p][W_PIN_IN] = brain->sumsi_stack;
            }
            break;
        case CMD_SUMSI_TO_WEIGHT_CTRL: // connect latest sumsi.out to weight[-ix].ctrl 
            // TODO Allow override?
            p = brain->weight_stack - ix;
            if(p >= 1) {
                brain->weight_conn[p][W_PIN_CTRL_TYPE] = TYPE_SUMSI_OUT;
                brain->weight_conn[p][W_PIN_CTRL] = brain->sumsi_stack;
            }
            break;     
        case CMD_WEIGHT_TO_SUMSI_IN: // connect latest weight.out to sumsi[-ix].in
            // TODO Allow override?
            p = brain->sumsi_stack - ix;
            if(p >= 1) {
                brain->weight_conn[brain->weight_stack][W_PIN_OUT_TYPE] = TYPE_SUMSI_IN;
                brain->weight_conn[brain->weight_stack][W_PIN_OUT] = p;
            }
            break;
        case CMD_WEIGHT_TO_WEIGHT_CTRL: // connect latest weight.out to weight[-ix].ctrl
            // TODO Allow override?
            p = brain->weight_stack - ix;
            if(p >= 1) {
                brain->weight_conn[brain->weight_stack][W_PIN_OUT_TYPE] = TYPE_WEIGHT_CTRL;
                brain->weight_conn[brain->weight_stack][W_PIN_OUT] = p;
                brain->weight_conn[p][W_PIN_CTRL_TYPE] = TYPE_WEIGHT_OUT;
                brain->weight_conn[p][W_PIN_CTRL] = brain->weight_stack;
            }
            break;        
        case CMD_POP_WEIGHT:
            if(brain->weight_stack > 1) { brain->weight_stack--; }
            break;
        case CMD_POP_SUMSI:
            if(brain->sumsi_stack > 1) { brain->sumsi_stack--; }
            break;
        case CMD_WEIGHT_TO_INPUT: // Connect the main input[ix] to the latest weight.in
            // TODO Allow override?
            // TODO Allow on non-main thread?
            if(ix < NUM_INPUTS) {
                brain->weight_conn[brain->weight_stack][W_PIN_IN_TYPE] = TYPE_GLOBAL_IN;
                brain->weight_conn[brain->weight_stack][W_PIN_IN] = ix;
                brain->input_conn[ix] = brain->weight_stack;
            }
            else {
                die("Overindexed input");
            }
            break;
        case CMD_SUMSI_TO_OUT: // Connect the latest sumsi.out to the main output
            // TODO Allow override?
            // TODO Allow on non-main thread?
            brain->output = brain->sumsi_stack;
        default:
            die("Unknown command");
    }
    
}




// Initialise a brain for play
void brain_play_init(const struct brain_t *brain) {
}

// Perform one step of thinking
void brain_play_step(const struct brain_t *brain) {
    
}
