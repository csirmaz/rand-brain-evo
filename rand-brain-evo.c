#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

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

#define NUM_INPUTS 9 // red_example(x,y), blue_example(x,y), question(x,y), energy, clock, bias
#define MAX_WEIGHTS 10000
#define MAX_SUMSIS 100
#define MAX_GENES 10000

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

#define TYPE_VALUE float

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
    TYPE_VALUE initial_weights[MAX_WEIGHTS];
    TYPE_VALUE weights[MAX_WEIGHTS];
    
    // For initial calculations
    TYPE_VALUE weight_state[MAX_WEIGHTS];
    TYPE_VALUE sumsi_state[MAX_SUMSIS];
    TYPE_VALUE input_state[NUM_INPUTS];
};

struct brain_t *brain_alloc(void) {
    struct brain_t *brain = malloc(sizeof(struct brain_t));
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
    brain->learning_rate = 1e-5;
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
void brain_play_step(struct brain_t *brain) {
    int i, p;
    TYPE_VALUE ctrl;

    // Update weight states (these will represent the inputs to the weights)
    for(i=1; i<=brain->weight_stack_max; i++) {
        p = brain->weight_conn[i][W_PIN_IN];
        if(p > 0) {
            switch(brain->weight_conn[i][W_PIN_IN_TYPE]) {
                case TYPE_GLOBAL_IN:
                    brain->weight_state[i] = brain->input_state[p];
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
    int commands[MAX_GENES];
    int args[MAX_GENES];
    int length;
};

struct genes_t *genes_alloc() {
    struct genes_t *genes = malloc(sizeof(struct genes_t));
    if(genes == NULL) { die("Out of memory"); }
    return genes;
}

void genes_init(struct genes_t *genes) {
    genes->learning_rate = 1e-2;
    
    genes->commands[0] = CMD_WEIGHT_TO_INPUT;
    genes->args[0] = 8;
    
    genes->commands[1] = CMD_WEIGHT_TO_SUMSI_IN;
    genes->args[1] = 0;
    
    genes->commands[2] = CMD_SUMSI_TO_OUT;
    genes->args[2] = 0; // dummy
    
    genes->length = 3;    
}

void genes_clone(const struct genes_t *source) {
    int i;
    struct genes_t *clone = genes_alloc();
    clone->learning_rate = source->learning_rate;
    for(i=0; i<source->length; i++) {
        clone->commands[i] = source->commands[i];
        clone->args[i] = source->args[i];
    }
}


struct brain_t *genes_create_brain(const struct genes_t *genes) {
    int i;
    struct brain_t *brain = brain_alloc();
    brain_constr_init(brain);
    brain->learning_rate = genes->learning_rate;
    for(i=0; i<genes->length; i++) {
        brain_constr_process_command(brain, genes->commands[i], genes->args[i]);
    }
    return brain;
}

void genes_mutate_learning_rate(struct genes_t *genes) {
    genes->learning_rate *= getrand() * .2 + .9;
}

void genes_mutate(struct genes_t *genes) {
    genes_mutate_learning_rate(genes);
    // TODO
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
#define TASK_EVAL_ZOOM 20.
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
// Evaluate a brain against a task. It needs to learn and respond

#define STEPS 500
#define STEPS_TO_ANSWER 40

// Return the energy of the brain (related to correct answers)
int evaluate(struct brain_t *brain, const struct task_t *task) {
    int target, answer, think, age=0;
    int energy = 0;
    
    brain_play_init(brain);
    brain->input_state[8] = 1.; // bias
    while(1) {
        task_get_question(
            task, 
            &brain->input_state[0], // pos_x
            &brain->input_state[1], 
            &brain->input_state[2], 
            &brain->input_state[3], 
            &brain->input_state[4], 
            &brain->input_state[5], // question_y
            &target
        );
        brain->input_state[6] = (TYPE_VALUE)energy;
        // Debug: task_plot(task, brain->input_state[0], brain->input_state[1], brain->input_state[2], brain->input_state[3], brain->input_state[4], brain->input_state[5]);
        
        for(think=0; think<STEPS_TO_ANSWER; think++) {
            brain->input_state[7] = ((TYPE_VALUE)think) / ((TYPE_VALUE)STEPS_TO_ANSWER); // clock
            brain_play_step(brain);
        }
        
        answer = (brain_get_output(brain) >= 0);
        if(answer == target) { energy++; } else { energy--; }
        // printf("Target: %d Answer: %d Energy: %d\n", target, answer, energy);
        age++;
        if(age > STEPS) { break; }
    }
    return energy;
}


// =======================================================================================================================

int main(void) {
    
    // See also https://linux.die.net/man/3/random_r
    srandom(time(NULL));
    
    struct genes_t *genes = genes_alloc();
    genes_init(genes);
    struct brain_t *brain = genes_create_brain(genes);
    struct task_t *task = task_alloc();
    task_init(task);
    printf("Ret: %d\n", evaluate(brain, task));    
    
    return 0;
}
