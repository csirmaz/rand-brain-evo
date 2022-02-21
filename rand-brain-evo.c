#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

// Configuration

// red_example(x,y), blue_example(x,y), question(x,y), energy, clock, bias
#define NUM_INPUTS 9
#define MAX_WEIGHTS 10000
#define MAX_SUMSIS 100
#define MAX_GENES 10000

#define TYPE_VALUE float
#define TYPE_VALUE_FORMAT "%f"
// define TYPE_VALUE double
// define TYPE_VALUE_FORMAT "%lf"

// Grid to evaluate task surfaces
#define TASK_EVAL_ZOOM 20.

#define INITIAL_LEARNING_RATE .1
#define INITIAL_THINKING_TIME 60
#define MIN_THINKING_TIME 6
#define MUTATE_THINKING_TIME 1

// How many questions to ask in a task (training/evaluation set)
#define STEPS 600

// How many genes-brains to maintain
#define POOL_SIZE 1024

// How many to keep for the next generation (at least half)
#define POOL_KEEP 680

// How many different tasks to give
#define TASK_NUM 3

// Penalise long sequences
#define GENE_LENGTH_PENALTY (((TYPE_VALUE)STEPS) * ((TYPE_VALUE)TASK_NUM) / ((TYPE_VALUE)MAX_WEIGHTS) / 2.)

// Penalise slow answers
#define THINKING_TIME_PENALTY (((TYPE_VALUE)STEPS) * ((TYPE_VALUE)TASK_NUM) / ((TYPE_VALUE)INITIAL_THINKING_TIME) / 20.)

// Configuration ends

// Commands are used to construct a network and form the gene sequences / threads
#define CMD_NEW_WEIGHT 901
#define CMD_NEW_SUMSI 902
#define CMD_SUMSI_TO_WEIGHT_IN 903
#define CMD_SUMSI_TO_WEIGHT_CTRL 904
#define CMD_WEIGHT_TO_SUMSI_IN 905
#define CMD_WEIGHT_TO_WEIGHT_CTRL 906
#define CMD_POP_WEIGHT 907
#define CMD_POP_SUMSI 908
#define CMD_WEIGHT_TO_INPUT 909
#define CMD_SUMSI_TO_OUT 910
#define CMD_CALL_THREAD 911 // TODO Not implemented yet

#define ARG_DUMMY -1
#define ARG_RAND_WEIGHT -2
#define ARG_RAND_SUMSI -3

// Indices when storing weight unit connections
#define W_PIN__NUM 6
#define W_PIN_OUT 0
#define W_PIN_OUT_TYPE 1
#define W_PIN_IN 2
#define W_PIN_IN_TYPE 3
#define W_PIN_CTRL 4
#define W_PIN_CTRL_TYPE 5

#define TYPE_WEIGHT_CTRL 801
#define TYPE_WEIGHT_OUT 802
#define TYPE_SUMSI_IN 803
#define TYPE_SUMSI_OUT 804
#define TYPE_GLOBAL_IN 805

// TODO Use double?
// TODO Allocate memory for brains instead of static lists
// TODO sanity check brain after creation (e.g. inputs and outputs are connected)


void die(char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}


// Returns a random number between 0 and 1
TYPE_VALUE getrand() {
    return ((TYPE_VALUE)random()) / ((TYPE_VALUE)RAND_MAX);
}


int getrand_location(const int length) {
    return (int)(getrand() * (length + 1));
}


void write_debug_file(char *msg) {
    FILE *outfile = fopen("debug.dat", "w+");
    if(outfile == NULL) { die("Cannot open file"); }
    fprintf(outfile, "%s", msg);
    fclose(outfile);
}


// ==== BRAIN ====================================================================================================================

struct brain_t {
    // Weight units have two inputs (input, control) and one output
    // We have a stack of weights for construction, but really they are always numbered sequentially, so we only need a number
    int weight_stack;
    int weight_stack_max;

    // Sumsi units (sum and sigmoid) can have many inputs and one output
    // We have a stack of them for construction, but really they are always numbered sequentially, so we only need a number
    int sumsi_stack;
    int sumsi_stack_max;
    
    // Outgoing connections from a weight unit
    // weight_conn[i][W_PIN_OUT] -- which whatever the output is going to
    // weight_conn[i][W_PIN_OUT_TYPE] -- whether the output is connected to a TYPE_WEIGHT_CTRL or a TYPE_SUMSI_IN
    // weight_conn[i][W_PIN_IN] -- which whatever the input is coming from
    // weight_conn[i][W_PIN_IN_TYPE] -- whether it is TYPE_GLOBAL_IN or TYPE_SUMSI_OUT
    // weight_conn[i][W_PIN_CTRL] -- which whatever the control is coming from
    // weight_conn[i][W_PIN_CTRL_TYPE] -- whether it is TYPE_WEIGHT_OUT or TYPE_SUMSI_OUT
    int weight_conn[MAX_WEIGHTS][W_PIN__NUM];
    
    // Which weights are inputs connected to? (For sanity checking)
    int input_conn[NUM_INPUTS];
    
    // Which sumsi does the output come from?
    int output_conn;
    
    TYPE_VALUE learning_rate;
    int thinking_time;
    TYPE_VALUE initial_weights[MAX_WEIGHTS];
    TYPE_VALUE weights[MAX_WEIGHTS];
    
    // For internal calculations
    TYPE_VALUE weight_state[MAX_WEIGHTS];
    TYPE_VALUE sumsi_state[MAX_SUMSIS];
};


struct brain_t *brain_alloc(int count) {
    struct brain_t *brain = malloc(count * sizeof(struct brain_t));
    if(brain == NULL) { die("Out of memory"); }
    return brain;
}


// Implements the nonlinearity that sumsis use
TYPE_VALUE nonlinearity(TYPE_VALUE x) {
    return (x < 0 ? x / 10. : x);
}


// Initialise a brain for construction
void brain_constr_init(struct brain_t *brain) {
    int i, j;
    brain->weight_stack = 1; // start with 1 so 0 can mean unconnected
    brain->initial_weights[1] = .5;
    brain->weight_stack_max = 1;
    brain->sumsi_stack = 1; // start with 1 so 0 can mean unconnected
    brain->sumsi_stack_max = 1;
    for(i=0; i<NUM_INPUTS; i++) { brain->input_conn[i] = 0; } // unconnected
    brain->output_conn = 0;
    for(i=0; i<MAX_WEIGHTS; i++) for(j=0; j<W_PIN__NUM; j++) brain->weight_conn[i][j] = 0;
    brain->learning_rate = 1e-5; // this is not relevant as overridden by (default) values in genes
    brain->thinking_time = 60; // this is not relevant as overridden by (default) values in genes
}


// Construction: process a command in the gene sequence
void brain_constr_process_command(struct brain_t *brain, int command, int ix) {
    int p;

    switch(command) {
        case CMD_NEW_WEIGHT: // create new weight unit and push. ix is the initial weight (ix/100)
            brain->weight_stack++;
            brain->initial_weights[brain->weight_stack] = ((TYPE_VALUE)ix) / 100;
            if(brain->weight_stack >= MAX_WEIGHTS) { die("Too many weights"); }
            if(brain->weight_stack > brain->weight_stack_max) { brain->weight_stack_max = brain->weight_stack; }
            break;
        case CMD_NEW_SUMSI: // create new sumsi unit and push
            brain->sumsi_stack++;
            if(brain->sumsi_stack >= MAX_SUMSIS) { die("Too many sumsis"); }
            if(brain->sumsi_stack > brain->sumsi_stack_max) { brain->sumsi_stack_max = brain->sumsi_stack; }
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
            brain->output_conn = brain->sumsi_stack;
            break;
        default:
            // printf("Command: %d", command);
            die("Unknown command");
    }
    
}


// Initialise a brain for thinking and learning
void brain_play_init(struct brain_t *brain) {
    int i;
    for(i=0; i<=brain->weight_stack_max; i++) { 
        brain->weight_state[i] = 0;
        brain->weights[i] = brain->initial_weights[i] + getrand() / 100.; // a bit of noise
    }
    for(i=0; i<=brain->sumsi_stack_max; i++) { brain->sumsi_state[i] = 0; }
}


// Perform one step of thinking and learning
void brain_play_step(struct brain_t *brain, TYPE_VALUE *input_state) {
    int i, p;
    TYPE_VALUE ctrl;

    // Update weight states (these will represent the inputs to the weights)
    for(i=1; i<=brain->weight_stack_max; i++) {
        p = brain->weight_conn[i][W_PIN_IN];
        if(p > 0) {
            switch(brain->weight_conn[i][W_PIN_IN_TYPE]) {
                case TYPE_GLOBAL_IN:
                    brain->weight_state[i] = input_state[p];
                    break;
                case TYPE_SUMSI_OUT:
                    brain->weight_state[i] = brain->sumsi_state[p];
                    break;
                default:
                    die("Unknown weight in type");
            }
        }
    }
    
    // Apply the weights
    for(i=1; i<=brain->weight_stack_max; i++) {
        brain->weight_state[i] *= brain->weights[i];
    }
    
    
    // Clear the sumsi states
    for(i=1; i<=brain->sumsi_stack_max; i++) {
        brain->sumsi_state[i] = 0;
    }
    
    // Calculate the sums in the sumsis
    for(i=1; i<=brain->weight_stack_max; i++) {
        p = brain->weight_conn[i][W_PIN_OUT];
        if(p > 0) {
            switch(brain->weight_conn[i][W_PIN_OUT_TYPE]) {
                case TYPE_SUMSI_IN:
                    brain->sumsi_state[p] += brain->weight_state[i];
                    break;
                case TYPE_WEIGHT_CTRL:
                    break;
                default:
                    die("Unknown weight out type");
            }
        }
    }
    
    // Apply the nonlinearity
    for(i=1; i<=brain->sumsi_stack_max; i++) {
        brain->sumsi_state[i] = nonlinearity(brain->sumsi_state[i]);
    }
    
    // Learning: apply the control
    for(i=1; i<=brain->weight_stack_max; i++) {
        p = brain->weight_conn[i][W_PIN_CTRL];
        if(p > 0) {
            switch(brain->weight_conn[i][W_PIN_CTRL_TYPE]) {
                case TYPE_WEIGHT_OUT:
                    ctrl = brain->weight_state[p];
                    break;
                case TYPE_SUMSI_OUT:
                    ctrl = brain->sumsi_state[p];
                    break;
                default:
                    die("Unknown weight ctrl type");
            }
            brain->weights[i] += ctrl * brain->learning_rate;
        }
    }
    
}


// Return the output from the brain
TYPE_VALUE brain_get_output(const struct brain_t *brain) {
    if(brain->output_conn == 0) { return 0; }
    return brain->sumsi_state[brain->output_conn];
}


// ==== GENES ====================================================================================================================

struct genes_t {
    TYPE_VALUE learning_rate;
    TYPE_VALUE thinking_time;
    int commands[MAX_GENES];
    int args[MAX_GENES];
    int length;
};


struct genes_t *genes_alloc(int count) {
    struct genes_t *genes = malloc(count * sizeof(struct genes_t));
    if(genes == NULL) { die("Out of memory"); }
    return genes;
}


// Initialise the genes
void genes_init(struct genes_t *genes) {
    genes->learning_rate = INITIAL_LEARNING_RATE;
    genes->thinking_time = INITIAL_THINKING_TIME;
    
    genes->commands[0] = CMD_WEIGHT_TO_INPUT;
    genes->args[0] = 8;
    
    genes->commands[1] = CMD_WEIGHT_TO_SUMSI_IN;
    genes->args[1] = 0;
    
    genes->commands[2] = CMD_SUMSI_TO_OUT;
    genes->args[2] = ARG_DUMMY;
    
    genes->length = 3;    
}


// Copy genes
void genes_clone(const struct genes_t *source, struct genes_t *clone) {
    clone->learning_rate = source->learning_rate;
    clone->thinking_time = source->thinking_time;
    for(int i=0; i<source->length; i++) {
        clone->commands[i] = source->commands[i];
        clone->args[i] = source->args[i];
    }
}


// Write genes into file
void genes_write(const struct genes_t *genes, FILE *fp, int human_readable) {
    if(!human_readable) { fprintf(fp, "brain_v1\n"); }
    fprintf(
        fp, 
        (human_readable ? "# learning_rate=%f steps=%f length=%d " : "%f\n%f\n%d\n"),
        genes->learning_rate, 
        genes->thinking_time, 
        genes->length
    );
    for(int i=0; i<genes->length; i++) {
        fprintf(
            fp, 
            (human_readable ? "[%d,%d] " : "%d\n%d\n"), 
            genes->commands[i], 
            genes->args[i]
        );
    }
    if(human_readable) { fprintf(fp, "\n"); }
}


// Load genes from file
void genes_read(struct genes_t *genes, FILE *fp) {
    size_t memlen = 0;
    char *membuf = NULL;
    int ret;
    int lineno = 0;
    int commandno;
    
    while(1) {
        ret = getline(&membuf, &memlen, fp);
        if(ret < 0) {
            free(membuf);
            die("Error while reading file");
        }
        if(membuf[0] == '#') { continue; }
        if(ret > 100) { die("Line too long"); }
        if(lineno == 0 && strcmp(membuf, "brain_v1\n") != 0) { printf("[%s]\n", membuf); die("Brain gene signature error"); }
        if(lineno == 1 && sscanf(membuf, TYPE_VALUE_FORMAT, &genes->learning_rate) != 1) { die("Brain gene error 2"); }
        if(lineno == 2 && sscanf(membuf, TYPE_VALUE_FORMAT, &genes->thinking_time) != 1) { die("Brain gene error 3"); }
        if(lineno == 3 && sscanf(membuf, "%d", &genes->length) != 1) { die("Brain gene error 4"); }
        if(lineno > 3) {
            commandno = (lineno - 4) / 2;
            if((lineno % 2) == 0) {
                if(sscanf(membuf, "%d", &genes->commands[commandno]) != 1) { die("Brain gene error 5"); }
                // printf("Loaded command L%d %d: %d\n", lineno, commandno, genes->commands[commandno]);
            }
            else {
                if(sscanf(membuf, "%d", &genes->args[commandno]) != 1) { die("Brain gene error 6"); }
                // printf("Loaded command arg L%d %d: %d\n", lineno, commandno, genes->args[commandno]);
                if(commandno == genes->length - 1) { break; }
            }
        }
        lineno++;
    }
    free(membuf);
}


// Create a brain based on the genes at the given location
// Tie down random offsets in the genes as we do so
void genes_create_brain(struct genes_t *genes, struct brain_t *brain) {
    int i;
    brain_constr_init(brain);
    brain->learning_rate = genes->learning_rate;
    brain->thinking_time = genes->thinking_time;
    for(i=0; i<genes->length; i++) {
        if(genes->args[i] == ARG_RAND_WEIGHT) {
            genes->args[i] = (int)(getrand() * (brain->weight_stack - 1));
        }
        else if(genes->args[i] == ARG_RAND_SUMSI) {
            genes->args[i] = (int)(getrand() * (brain->sumsi_stack - 1));
        }
        brain_constr_process_command(brain, genes->commands[i], genes->args[i]);
    }
}


// Inject a command at location
void genes_inject(struct genes_t *genes, const int location, const int command, const int arg) {
    if(location > genes->length) { die("Inject over end"); }
    if(genes->length >= MAX_GENES) { die("Genes too long"); }
    if(location < genes->length) {
        for(int i=genes->length; i>location; i--) {
            genes->commands[i] = genes->commands[i-1];
            genes->args[i] = genes->args[i-1];
        }
    }
    genes->commands[location] = command;
    genes->args[location] = arg;
    genes->length++;
}


// Remove a command at location
void genes_remove(struct genes_t *genes, const int location) {
    if(genes->length <= 1) { return; }
    genes->length--;
    for(int i=location; i<genes->length; i++) {
        genes->commands[i] = genes->commands[i+1];
        genes->args[i] = genes->args[i+1];
    }
}


// Mutate a gene sequence
void genes_mutate(struct genes_t *genes) {
#if MUTATE_THINKING_TIME
    static int modes_length = 13;
    static TYPE_VALUE modes[13] = {
#else
    static int modes_length = 12;
    static TYPE_VALUE modes[12] = {
#endif
        1, // 0: mutate learning rate
        1, // 1: inject CMD_SUMSI_TO_OUT
        2, // 2: inject CMD_POP_WEIGHT
        2, // 3: inject CMD_POP_SUMSI
        1, // 4: inject CMD_WEIGHT_TO_INPUT with random input
        7, // 5: remove command
        1, // 6: inject CMD_SUMSI_TO_WEIGHT_IN to random weight unit
        1, // 7: inject CMD_SUMSI_TO_WEIGHT_CTRL to random weight unit
        1, // 8: inject CMD_WEIGHT_TO_WEIGHT_CTRL to random weight unit
        1, // 9: inject CMD_WEIGHT_TO_SUMSI_IN to random sumsi unit
        2, // 10: new sumsi & connect to last weight
        2 // 11: new weight & connect to last sumsi
#if MUTATE_THINKING_TIME
        ,1  // 12: mutate thinking time
#endif
    };
    static int modes_init = 0;
    
    // Create probability boundaries
    if(modes_init == 0) {
        modes_init = 1;
        int i;
        TYPE_VALUE s = 0;
        for(i=0; i<modes_length; i++) { 
            s += modes[i];
            modes[i] = s;
        }
        for(i=0; i<modes_length; i++) { modes[i] /= s; /* printf("MODE %d = %f\n",i,modes[i]); */ }
    }
    
    TYPE_VALUE mode_v = getrand();
    int mode = 0, loc;
    while(modes[mode] < mode_v && mode < modes_length) { mode++; }
    // printf("Mutate mode_v: %f mode: %d\n", mode_v, mode);
    switch(mode) {
        case 0: // mutate learning rate
            genes->learning_rate *= getrand() * .4 + .8; break;
        case 1:
            genes_inject(genes, getrand_location(genes->length), CMD_SUMSI_TO_OUT, ARG_DUMMY); break;
        case 2:
            genes_inject(genes, getrand_location(genes->length), CMD_POP_WEIGHT, ARG_DUMMY); break;
        case 3:
            genes_inject(genes, getrand_location(genes->length), CMD_POP_SUMSI, ARG_DUMMY); break;
        case 4:
            genes_inject(genes, getrand_location(genes->length), CMD_WEIGHT_TO_INPUT, ((int)(getrand() * NUM_INPUTS))); break;
        case 5:
            genes_remove(genes, ((int)(getrand() * genes->length))); break;
        case 6:
            genes_inject(genes, getrand_location(genes->length), CMD_SUMSI_TO_WEIGHT_IN, ARG_RAND_WEIGHT); break;
        case 7:
            genes_inject(genes, getrand_location(genes->length), CMD_SUMSI_TO_WEIGHT_CTRL, ARG_RAND_WEIGHT); break;
        case 8:
            genes_inject(genes, getrand_location(genes->length), CMD_WEIGHT_TO_WEIGHT_CTRL, ARG_RAND_WEIGHT); break;
        case 9:
            genes_inject(genes, getrand_location(genes->length), CMD_WEIGHT_TO_SUMSI_IN, ARG_RAND_SUMSI); break;
        case 10:
            loc = getrand_location(genes->length);
            genes_inject(genes, loc, CMD_NEW_SUMSI, ARG_DUMMY);
            genes_inject(genes, loc+1, CMD_WEIGHT_TO_SUMSI_IN, 0);
            break;
        case 11:
            loc = getrand_location(genes->length);
            genes_inject(genes, loc, CMD_NEW_WEIGHT, (int)(getrand() * 200. - 100.));
            genes_inject(genes, loc+1, CMD_SUMSI_TO_WEIGHT_IN, 0);
            break;
#if MUTATE_THINKING_TIME
        case 12:
            genes->thinking_time *= getrand() * .4 + .8; 
            if(genes->thinking_time < MIN_THINKING_TIME) { genes->thinking_time = MIN_THINKING_TIME; }
            break;
#endif
        default:
            die("Unknown mutation mode");
    }
}


void genes_crossover(const struct genes_t *src1, const struct genes_t *src2, struct genes_t *dst1, struct genes_t *dst2) {
    TYPE_VALUE start, end, snip;
    int start1, start2, end1, end2, i, a, b;
    
    snip = getrand() * .8; // length of snippet (0..1)
    start = getrand() * (1. - snip); // starting point (0..1)
    end = start + snip;

    start1 = src1->length * start;    
    start2 = src2->length * start;    
    end1 = src1->length * end;
    end2 = src2->length * end;
    
    // printf("XO snip %f start %f end %f start1 %d end1 %d L1 %d start2 %d end2 %d L2 %d\n",snip,start,end,start1,end1,src1->length,start2,end2,src2->length);
    
    // src1:  0 a1a1a1 start1 b1b1b1 end1 c1c1c1
    // src2:  0 a2a2a2 start2 b2b2b2 end2 c2c2c2
    
    // dst1:  0 a1a1a1 start1 b2b2b2 (start1+end2-start2) c1c1c1
    // dst2:  0 a2a2a2 start2 b1b1b1 (start2+end1-start1) c2c2c2
    
    dst1->learning_rate = src1->learning_rate * (1. - snip) + src2->learning_rate * snip;
    dst1->thinking_time = src1->thinking_time * (1. - snip) + src2->thinking_time * snip;
    dst1->length = start1 + (end2 - start2) + (src1->length - end1);
    if(dst1->length >= MAX_GENES) { die("Crossover too long 1"); }
    for(i=0; i<start1; i++) {
        a = i;
        b = i;
        if(a >= dst1->length) { die("Crossover E1"); }
        if(b >= src1->length) { die("Crossover E2"); }
        dst1->commands[a] = src1->commands[b]; 
        dst1->args[a]     = src1->args[b]; 
    }
    for(i=0; i<(end2-start2); i++) { 
        a = i + start1;
        b = i + start2;
        if(a >= dst1->length) { die("Crossover E3"); }
        if(b >= src2->length) { die("Crossover E4"); }
        dst1->commands[a] = src2->commands[b]; 
        dst1->args[a]     = src2->args[b]; 
    }
    for(i=0; i<(src1->length-end1); i++) {
        a = i+start1+end2-start2;
        b = i+end1;
        if(a >= dst1->length) { die("Crossover E5"); }
        if(a < 0) { die("Crossover E6"); }
        if(b >= src1->length) { die("Crossover E7"); }
        dst1->commands[a] = src1->commands[b];
        dst1->args[a]     = src1->args[b];
    }
    
    dst2->learning_rate = src2->learning_rate * (1. - snip) + src1->learning_rate * snip;
    dst2->thinking_time = src2->thinking_time * (1. - snip) + src1->thinking_time * snip;
    dst2->length = start2 + (end1 - start1) + (src2->length - end2);
    if(dst2->length >= MAX_GENES) { die("Crossover too long 2"); }
    for(i=0; i<start2; i++) { 
        a = i;
        b = i;
        if(a >= dst2->length) { die("Crossover E8"); }
        if(b >= src2->length) { die("Crossover E9"); }
        dst2->commands[a] = src2->commands[b]; 
        dst2->args[a]     = src2->args[b]; 
    }
    for(i=0; i<(end1-start1); i++) { 
        a = i + start2;
        b = i + start1;
        if(a >= dst2->length) { die("Crossover E10"); }
        if(b >= src1->length) { die("Crossover E11"); }
        dst2->commands[a] = src1->commands[b]; 
        dst2->args[a]     = src1->args[b]; 
    }
    for(i=0; i<(src2->length-end2); i++) { 
        a = i+start2+end1-start1;
        b = i+end2;
        if(a >= dst2->length) { die("Crossover E12"); }
        if(a < 0) { die("Crossover E13"); }
        if(b >= src2->length) { die("Crossover E14"); }
        dst2->commands[a] = src2->commands[b];
        dst2->args[a]     = src2->args[b];
    }
}

// ==== TASK ===================================================================================================================
// Create a wavy surface

struct task_t {
    TYPE_VALUE x_freq1;
    TYPE_VALUE x_phase1;

    TYPE_VALUE y_freq1;
    TYPE_VALUE y_phase1;

    TYPE_VALUE x_freq2;
    TYPE_VALUE x_phase2;

    TYPE_VALUE y_freq2;
    TYPE_VALUE y_phase2;
    
    TYPE_VALUE pol_freq;
    TYPE_VALUE pol_phase;
};

TYPE_VALUE task_init_freq(void) {
    TYPE_VALUE min_freq = .2;
    TYPE_VALUE max_freq = 6.;
    return min_freq + getrand() * (max_freq - min_freq);    
}

TYPE_VALUE task_init_phase(void) {
    return getrand() * 3.14159;
}

struct task_t *task_alloc(void) {
    struct task_t *task = malloc(sizeof(struct task_t));
    if(task == NULL) { die("Out of memory"); }
    return task;
}

// Get a value on the surface. Call with x, y in [-1, 1]
TYPE_VALUE task_get_value(const struct task_t *task, TYPE_VALUE x, TYPE_VALUE y) {
    TYPE_VALUE o = 0, pol;
    o += sin(task->x_freq1 * x + task->x_phase1);
    o += sin(task->y_freq1 * y + task->y_phase1);
    o += sin(task->x_freq2 * x + task->x_phase2);
    o += sin(task->y_freq2 * y + task->y_phase2);
    pol = sqrt(x*x + y*y);
    o += sin(task->pol_freq * pol + task->pol_phase);
    return o;
}


// Ensure the positive and negative areas are roughly equal so the test set (the questions) would be evenly distributed
int task_evaluate(const struct task_t *task) {
    int i, j, num_pos=0, num_neg=0;
    TYPE_VALUE x, y, v;
    
    for(i=0; i<TASK_EVAL_ZOOM*2.; i++) {
        x = (i / TASK_EVAL_ZOOM) - 1.;
        for(j=0; j<TASK_EVAL_ZOOM*2.; j++) {
            y = (j / TASK_EVAL_ZOOM) - 1.;
            v =task_get_value(task, x, y);
            if(v > 0) { num_pos++; }
            if(v < 0) { num_neg++; }
        }
    }
    return abs(num_pos - num_neg) < (TASK_EVAL_ZOOM * TASK_EVAL_ZOOM * .05);
}


// Initialise the surface
void task_init(struct task_t *task) {
    while(1) {
        task->x_freq1 = task_init_freq();
        task->y_freq1 = task_init_freq();
        task->x_freq1 = task_init_freq();
        task->y_freq2 = task_init_freq();
        task->pol_freq = task_init_freq();
        task->x_phase1 = task_init_phase();
        task->x_phase1 = task_init_phase();
        task->y_phase2 = task_init_phase();
        task->y_phase2 = task_init_phase();
        task->pol_phase = task_init_phase();
        if(task_evaluate(task)) { break; }
    }
}


// Print a plot of a task (surface)
void task_plot(
    const struct task_t *task,
    const TYPE_VALUE x1,
    const TYPE_VALUE y1,
    const TYPE_VALUE x2,
    const TYPE_VALUE y2,
    const TYPE_VALUE x3,
    const TYPE_VALUE y3
) {
    int i, j;
    char c;
    TYPE_VALUE zoom = 20., x, y;
    
    for(i=0; i<zoom*2.; i++) {
        x = (i / zoom) - 1.;
        for(j=0; j<zoom*2.; j++) {
            y = (j / zoom) - 1.;
            c = (task_get_value(task, x, y) >= 0 ? '#' : '-');
            printf("%c", c);
            if(fabs(x-x1) < .05 && fabs(y-y1) < .05) { c = '1'; }
            if(fabs(x-x2) < .05 && fabs(y-y2) < .05) { c = '2'; }
            if(fabs(x-x3) < .05 && fabs(y-y3) < .05) { c = 'Q'; }
            printf("%c", c);
        }
        printf("\n");
    }
}


// Test the task code by printing sample surfaces
void task_test(void) {
    struct task_t *task = task_alloc();
    task_init(task);
    task_plot(task, 0,0,0,0,0,0);
}


// Get a random coordinate value
TYPE_VALUE task_get_coord(void) {
    return getrand() * 2. - 1.;
}


// Get a training question from a task
// This returns the coordinates of a positive point, the coordinates of a negative point, a question point and a target answer
void task_get_question(
    const struct task_t *task,
    TYPE_VALUE *pos_x, 
    TYPE_VALUE *pos_y,
    TYPE_VALUE *neg_x,
    TYPE_VALUE *neg_y,
    TYPE_VALUE *question_x,
    TYPE_VALUE *question_y,
    int *target
) {
    TYPE_VALUE x, y;
    int no_pos = 1, no_neg = 1;
    while(no_pos || no_neg) {
        x = task_get_coord();
        y = task_get_coord();
        if(task_get_value(task, x, y) < 0) {
            no_neg = 0;
            *neg_x = x;
            *neg_y = y;
        }
        else {
            no_pos = 0;
            *pos_x = x;
            *pos_y = y;
        }
    }
    x = task_get_coord();
    y = task_get_coord();
    *question_x = x;
    *question_y = y;
    *target = (task_get_value(task, x, y) >= 0);
}


// ==== EVALUATE ===================================================================================================================

// Evaluate brains against a task. They need to learn and respond
// Return the energy of the brain (related to correct answers)
int evaluate(struct brain_t *brainpool, const struct task_t *task, TYPE_VALUE *results, int best_brain) {
    int target, answer, think, i, question_num;
    TYPE_VALUE thinking_time_v;
    TYPE_VALUE input_state[NUM_INPUTS];
    int target_1_num = 0, best_brain_1_num = 0, best_brain_correct_num = 0; // stats
    
    input_state[8] = 1.; // bias
    
    for(i=0; i<POOL_SIZE; i++) {
        brain_play_init(&brainpool[i]);
        // results[i] = 0; -- initialised elsewhere
    }
    
    for(question_num=0; question_num<STEPS; question_num++) { // Loop through questions
        
        task_get_question(
            task, 
            &input_state[0], // pos_x
            &input_state[1], 
            &input_state[2], 
            &input_state[3], 
            &input_state[4], 
            &input_state[5], // question_y
            &target
        );
        
        if(target) { target_1_num++; } // stats
        
        for(i=0; i<POOL_SIZE; i++) { // Loop through brains
            
            input_state[6] = results[i];
            // Debug: task_plot(task, brain->input_state[0], brain->input_state[1], brain->input_state[2], brain->input_state[3], brain->input_state[4], brain->input_state[5]);
        
            thinking_time_v = brainpool[i].thinking_time;
            for(think = 0; think < thinking_time_v; think++) { // Loop thinking
                input_state[7] = ((TYPE_VALUE)think) / thinking_time_v; // clock
                brain_play_step(&brainpool[i], input_state);
            }
        
            answer = (brain_get_output(&brainpool[i]) >= 0);
            if(answer == target) { results[i]++; } else { results[i]--; }
            // if(i == best_brain) { printf("Best brain: %d Question: %d Answer: %d Target: %d Result: %f\n", i, question_num, answer, target, results[i]); }
            if(i == best_brain) { 
                if(answer) { best_brain_1_num++; }
                if(answer == target) { best_brain_correct_num++; }
            }
            
        } // end loop through brains
    } // end loop through questions
    
    fprintf(stderr, "Task: Prev best brain: %d Target=1ratio: %f Answer=1ratio: %f CorrectRatio: %f\n", best_brain, ((TYPE_VALUE)target_1_num) / STEPS, ((TYPE_VALUE)best_brain_1_num) / STEPS, ((TYPE_VALUE)best_brain_correct_num) / STEPS);
}


// =======================================================================================================================

// Compare numbers
static int cmpint(const void *p1, const void *p2) { return ( *((TYPE_VALUE*)p1) > *((TYPE_VALUE*)p2) ) - ( *((TYPE_VALUE*)p1) < *((TYPE_VALUE*)p2) ); }


void dump_genepool(struct genes_t *genepool) {
    fprintf(stderr, "Writing gene pool to file...\n");
    FILE *outfile = fopen("genepool.dat", "w+");
    if(outfile == NULL) { die("Cannot open file"); }
    fprintf(outfile, "genepool_v1\n# Pool size:\n%d\n", POOL_SIZE);
    for(int i=0; i<POOL_SIZE; i++) { genes_write(&genepool[i], outfile, 1); }
    for(int i=0; i<POOL_SIZE; i++) { genes_write(&genepool[i], outfile, 0); }
    fclose(outfile);
}


void load_genepool(struct genes_t *genepool) {
    fprintf(stderr, "Loading gene pool from file...\n");
    FILE *outfile = fopen("genepool.dat", "r");
    if(outfile == NULL) { die("Cannot open file"); }
    size_t memlen = 0;
    char *membuf = NULL;
    int ret;
    int lineno = 0;
    int pool_size;
    
    while(1) {
        ret = getline(&membuf, &memlen, outfile);
        if(ret < 0) {
            free(membuf);
            die("Error while reading file");
        }
        if(membuf[0] == '#') { continue; }
        if(ret > 100) { die("Line too long"); }
        if(lineno == 0 && strcmp(membuf, "genepool_v1\n") != 0) { die("Genepool signature error"); }
        if(lineno == 1) {
            if(sscanf(membuf, "%d", &pool_size) != 1) { die("Genepool error 2"); }
            if(pool_size != POOL_SIZE) { die("Pool size mismatch"); }
            break;
        }
        lineno++;
    }
    for(int i=0; i<POOL_SIZE; i++) { genes_read(&genepool[i], outfile); }
    free(membuf);
    fclose(outfile);
    fprintf(stderr, "Loading gene pool from file done.\n");
}


// Usage: $0 [new]
int main(int argc, char **argv) {
    int p_load_genes = 1;
    int i, j, evo_steps=0;
    struct task_t *task;
    
    if(argc == 2 && strcmp(argv[1], "new") == 0) { p_load_genes = 0; }
    
    // See also https://linux.die.net/man/3/random_r
    srandom(time(NULL));
    
    struct genes_t *genepool;
    genepool = genes_alloc(POOL_SIZE);
    struct brain_t *brainpool;
    brainpool = brain_alloc(POOL_SIZE);
    
    if(p_load_genes) {
        load_genepool(genepool);
    }
    else {
        fprintf(stderr, "Initializing new gene pool...\n");
        for(i=0; i<POOL_SIZE; i++) {
            genes_init(&genepool[i]);
            genes_mutate(&genepool[i]);
            // genes_print(&genepool[i]);
        }
    }
        
    for(i=0; i<POOL_SIZE; i++) {
        genes_create_brain(&genepool[i], &brainpool[i]);
    }
    
    TYPE_VALUE results[POOL_SIZE];
    TYPE_VALUE results2[POOL_SIZE];
    task = task_alloc();
    int best_brain = -1;
    
    while(1) {
        
        // Initialise results array
        for(i=0; i<POOL_SIZE; i++) {
            // We add a bit of randomness because there are too many results that are the same
            results[i] = 0 - GENE_LENGTH_PENALTY * (genepool[i].length + getrand() / 2.) - THINKING_TIME_PENALTY * genepool[i].thinking_time;
        }
        if(best_brain != -1) { fprintf(stderr, "Best brain: %d Length: %d Thinking time: %f Initial score: %f Length penalty: %f Time penalty: %f\n", best_brain, genepool[best_brain].length, genepool[best_brain].thinking_time, results[best_brain], GENE_LENGTH_PENALTY, THINKING_TIME_PENALTY); }
        
        for(j=0; j<TASK_NUM; j++) {
            // Create a new task
            task_init(task);
            // Give the task to the brains
            evaluate(brainpool, task, results, best_brain);
        }
    
        // Order the brains - best LAST!
        // worst                                               best
        // |------------------------------------------------------|
        // 0                                              POOL_SIZE
        //                              |--- SIZE - KEEP ---------| top ones
        //                    |------- POOL_KEEP -----------------| keep me
        for(i=0; i<POOL_SIZE; i++) {
            results2[i] = results[i];
        }
        qsort(results2, POOL_SIZE, sizeof(TYPE_VALUE), cmpint);
        TYPE_VALUE best_value = results2[POOL_SIZE - 1];
        TYPE_VALUE top_limit_value = results2[POOL_KEEP + 2]; // selects the top POOL_SIZE - POOL_KEEP - 2 many (keep 2 for the crossover)
        TYPE_VALUE limit_value = results2[POOL_SIZE - POOL_KEEP];
        fprintf(stderr,
            "Best: %f=%f%% at %d Top limit: %f = %f%% at %d Keep limit: %f=%f%% at %d\n",
            best_value,
            best_value / STEPS / TASK_NUM * 100.,
            POOL_SIZE - 1,
            top_limit_value,
            top_limit_value / STEPS / TASK_NUM * 100.,
            POOL_KEEP + 2,
            limit_value,
            limit_value / STEPS / TASK_NUM * 100.,
            POOL_SIZE - POOL_KEEP
        );
        write_debug_file("00best");
        
        // Now clone and mutate the top performers (POOL_SIZE - POOL_KEEP many)
        // There may be some edge cases, but whatever.
        int source_ix = 0;
        int target_ix = 0;
        int cloned = 0;
        best_brain = -1;
        int crossover_target[2];
        for(i=0; i<POOL_SIZE; i++) {
            if(results[i] == best_value) { best_brain = i; break; }
        }
        write_debug_file("10preloop");
        while(1) {
            write_debug_file("19loop");
            while(results[target_ix] >= limit_value && target_ix < POOL_SIZE) { target_ix++; }
            while(results[source_ix] < top_limit_value && source_ix < POOL_SIZE) { source_ix++; }
            if(source_ix >= POOL_SIZE || target_ix >= POOL_SIZE) { break; }
            
            if(cloned < 2) {
                crossover_target[cloned] = target_ix;
                cloned++;
                target_ix++;
                continue;
            }
            write_debug_file("20clone");
            
            // printf("Copying %d (res %f) to %d (res %f)\n", source_ix, results[source_ix], target_ix, results[target_ix]);
            genes_clone(&genepool[source_ix], &genepool[target_ix]);
            write_debug_file("22mutate1");
            genes_mutate(&genepool[target_ix]);
            write_debug_file("24mutate2");
            // Randomly, mutate twice
            if(getrand() < .5) { genes_mutate(&genepool[target_ix]); }
            write_debug_file("26brain");
            // Regenerate brain
            genes_create_brain(&genepool[target_ix], &brainpool[target_ix]);
            write_debug_file("28finish");
            source_ix++;
            target_ix++;
            cloned++;
            write_debug_file("29endloop");
        }
        write_debug_file("40cloned");
        fprintf(stderr, "Cloned: %d\n", cloned);
        
        // Crossover
        // Now we have reasonably good brains
        int crossover_source;
        while(1) {
            crossover_source = getrand() * POOL_SIZE;
            if(crossover_source != best_brain && crossover_source != crossover_target[0] && crossover_source != crossover_target[1]) { break; }
        }
        fprintf(stderr, "Crossover %d, %d -> %d, %d\n", best_brain, crossover_source, crossover_target[0], crossover_target[1]);
        write_debug_file("50costart");
        genes_crossover(&genepool[best_brain], &genepool[crossover_source], &genepool[crossover_target[0]], &genepool[crossover_target[1]]);
        write_debug_file("60coend");
        
        if((evo_steps % 20) == 0) { dump_genepool(genepool); }
        evo_steps++;
        write_debug_file("999endloop");
    }
        
    return 0;
}
