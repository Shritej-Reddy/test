
<?php
	
	$param = $_GET["usertext"];
	//$text = json_encode($param) ;
    echo gettype($param);
	$answer = $_GET['radiogroup'];  
if ($answer == "URL") { 
	echo "Using URL";         
    $command = escapeshellcmd('python keywordextraction.py --url ' .$param);
    $output = shell_exec($command);  
    //echo $output;   
}
else {
	echo "Using text";
    $command = escapeshellcmd('python keywordextraction.py --text ' .$param);
    $output = shell_exec($command);
    //echo $output;
}       

    header("location:dbcon1.php");

?>

