from streamlit_app import extract_price_from_text, is_price_query


def run_tests():
    # is_price_query
    assert is_price_query('What is the last traded price of TATA MOTORS today?')
    assert is_price_query('What is the current stock price of LSEG?')
    assert not is_price_query("What are Microsoft's AI plans?")

    # extract with symbol
    text = 'The stock is trading at $123.45 as of now.'
    res = extract_price_from_text(text)
    assert res is not None, 'Failed to extract $123.45'
    assert float(res['price']) == 123.45
    assert res['currency'] == 'USD'

    # extract with currency word
    text = 'Latest quote: 1,234.56 rupees per share.'
    res = extract_price_from_text(text)
    assert res is not None, 'Failed to extract 1,234.56 rupees'
    assert float(res['price']) == 1234.56

    # trading at phrase
    text = 'Stock is trading at 567.89 currently.'
    res = extract_price_from_text(text)
    assert res is not None, 'Failed to extract 567.89'
    assert float(res['price']) == 567.89

    print('All tests passed!')


if __name__ == '__main__':
    run_tests()
