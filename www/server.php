<?php

// Runs behind a webserver and stores and responds with gene sequences for xpol (cross-pool communication)

$PATH = dirname(__FILE__);
require $PATH . '/sqlbrite/sqlbrite.php';

$db = new SQLBrite(new SQLite3($PATH.'/data/server.sqlite'));

if(isset($_POST['todo'])) {
    if($_POST['todo'] == 'get') {
        $data = $db->querysingle('select genes from xpolgenes where created > datetime("now", "-1 hour") order by random() limit 1');
        print($data);
        exit;
    }
    if($_POST['todo'] == 'put' && isset($_POST['data'])) {
        $db->exec('insert into xpolgenes (poolid, genes) values (?, ?)', [0, $_POST['data']]);
        $db->exec('delete from xpolgenes where created < datetime("now", "-1 hour")');
        print('ok');
        exit;
    }    
}

http_response_code(500);

?>