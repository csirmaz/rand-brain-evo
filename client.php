<?php

// Launches the rand-brain-evo binary and manages it, and handles web requests for xpol (cross-pool communication)

require('config.php');
$running = true;
$state = 'noop';

function lg($msg) { fwrite(STDERR, $msg); }

function sighandler($signo) {
    global $running, $state;
    switch($signo) {
        case SIGTERM:
            lg("PHP: SIGTERM\n");
            $running = false;
            break;
        case SIGINT:
            lg("PHP: SIGINT\n");
            $running = false;
            break;
        case SIGUSR1:
            lg("PHP: SIGUSR1 (download)\n");
            $state = 'download';
            break;
        case SIGUSR2:
            lg("PHP: SIGUSR2 (upload)\n");
            $state = 'upload';
            break;
    }
}

function post($data) {
    global $p_url, $running;
    // $data = array('key1' => 'value1', 'key2' => 'value2');

    $options = array(
        'http' => array(
            'header'  => "Content-type: application/x-www-form-urlencoded\r\n",
            'method'  => 'POST',
            'content' => http_build_query($data)
        )
    );
    $context  = stream_context_create($options);
    $result = file_get_contents($p_url, false, $context);
    if ($result === FALSE) { lg("PHP request error\n"); $running = false; }
    return $result;
}

$parentpid = getmypid();
$childpid = pcntl_fork();
if($childpid == -1) { lg("PHP: Cannot fork\n"); exit(1); }
if($childpid == 0) {
    # Child code
    pcntl_exec($p_exec, [$parentpid]);
    exit(0);
}
# Parent code
pcntl_async_signals(true);
pcntl_signal(SIGTERM, "sighandler");
pcntl_signal(SIGINT, "sighandler");
pcntl_signal(SIGUSR1, "sighandler");
pcntl_signal(SIGUSR2, "sighandler");
lg("PHP: Parent PID: $parentpid Child PID: $childpid\n");

while($running) { 
    sleep(1);
    if($state == 'download') {
        lg("PHP: Downloading\n");
        $data = post(['todo'=>'get']);
        file_put_contents($p_file, $data);
        posix_kill($childpid, SIGUSR1);
    }
    if($state == 'upload') {
        lg("PHP: Uploading\n");
        $data = file_get_contents($p_file);
        post(['todo'=>'put', 'data'=>$data]);
        posix_kill($childpid, SIGUSR2);
    }
    $state = 'noop';
}

lg("PHP: Sending TERM to child\n");
posix_kill($childpid, SIGTERM); 
pcntl_wait($status);

?>