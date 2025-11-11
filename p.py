# =============================================================================
# STEP 1: IMPORTS AND CONFIGURATION
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import re
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Product Price Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
CONFIG = {
    'data_path': r"mon price bf 19k - TorobPay - Final - 11 Nov - 100k.csv",
    'colors': {
        'primary': '#3498db',
        'secondary': '#2c3e50',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#34495e',
        'light': '#ecf0f1',
        'dark': '#34495e'
    },
    'chart_templates': 'plotly_white'
}


# =============================================================================
# STEP 2: DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================

def load_and_preprocess_data():
    """Load and preprocess data with proper product frequency handling"""
    try:
        if os.path.exists(CONFIG['data_path']):
            df = pd.read_csv(CONFIG['data_path'])
            st.success(f"Data loaded successfully: {len(df):,} rows")
        else:
            st.warning("CSV file not found, creating sample data...")
            df = create_sample_data()

        # Clean column names
        df.columns = df.columns.str.strip()

        # Essential columns check and cleaning
        df = clean_and_validate_data(df)

        # Add analytical columns
        df = add_price_analysis_columns(df)

        st.info(f"Final dataset: {len(df):,} rows, {df['MON'].nunique():,} unique products")
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()


def create_sample_data():
    """Create realistic sample data"""
    np.random.seed(42)

    sample_mon_codes = ['MON_00123', 'MON_00456', 'MON_00789', 'MON_01234', 'MON_05678']
    sample_product_names = [
        'Xiaomi Redmi Note 12 128GB',
        'Samsung Galaxy A54 5G 256GB',
        'Apple iPhone 14 Pro 128GB',
        'Huawei P60 Pro 512GB',
        'OnePlus 11 5G 256GB'
    ]

    data = []
    for mon, product_name in zip(sample_mon_codes, sample_product_names):
        num_sellers = np.random.randint(5, 12)
        base_price = np.random.randint(800000, 3000000)

        for i in range(num_sellers):
            price_variation = np.random.uniform(0.8, 1.3)
            price = int(base_price * price_variation)

            data.append({
                'MON': mon,
                'Product Titles': product_name,
                'Product Prices': price,
                'Seller': f'Seller_{i + 1:02d}',
                'Platform': np.random.choice(['SnappPay', 'Torob', 'Marketplace_C'], p=[0.4, 0.4, 0.2])
            })

    df = pd.DataFrame(data)
    st.info(f"Sample data created: {len(df):,} rows, {df['MON'].nunique():,} unique products")
    return df


def clean_and_validate_data(df):
    """Clean and validate the dataframe"""
    st.write("Cleaning and validating data...")

    # Ensure required columns exist
    required_columns = ['MON', 'Product Titles', 'Product Prices', 'Seller']
    for col in required_columns:
        if col not in df.columns:
            st.warning(f"Missing column {col}")
            if col == 'Product Prices':
                df[col] = np.random.randint(1000000, 5000000, len(df))
            elif col == 'Seller':
                df[col] = [f'Seller_{i:02d}' for i in range(1, len(df) + 1)]
            elif col == 'Product Titles':
                df[col] = 'Unknown Product'

    # Clean data
    initial_count = len(df)
    df = df.dropna(subset=['MON', 'Product Prices'])
    df['Product Prices'] = pd.to_numeric(df['Product Prices'], errors='coerce')
    df = df.dropna(subset=['Product Prices'])
    st.write(f"Removed {initial_count - len(df)} rows with missing/invalid data")

    # Add Platform if missing
    if 'Platform' not in df.columns:
        platforms = ['SnappPay', 'Torob', 'Marketplace_C']
        df['Platform'] = np.random.choice(platforms, len(df), p=[0.4, 0.4, 0.2])
        st.write("Added missing Platform column")

    return df


def add_price_analysis_columns(df):
    """Add analytical columns for price analysis"""
    st.write("Adding analytical columns...")

    # Calculate price statistics per MON (product)
    product_stats = df.groupby('MON')['Product Prices'].agg([
        'min', 'max', 'mean', 'std', 'count'
    ]).reset_index()
    product_stats.columns = ['MON', 'min_price', 'max_price', 'mean_price', 'std_price', 'seller_count']

    df = df.merge(product_stats, on='MON', how='left')

    # Calculate analytical metrics
    df['price_vs_avg'] = df['Product Prices'] - df['mean_price']
    df['price_vs_min'] = df['Product Prices'] - df['min_price']
    df['savings_opportunity'] = df['Product Prices'] - df['min_price']
    df['is_cheapest'] = df['Product Prices'] == df['min_price']
    df['price_ratio'] = (df['Product Prices'] / df['mean_price'] - 1) * 100

    # Add price segments
    def get_price_segment(row):
        if row['Product Prices'] == row['min_price']:
            return 'Cheapest'
        elif row['Product Prices'] <= row['mean_price']:
            return 'Below Average'
        else:
            return 'Above Average'

    df['price_segment'] = df.apply(get_price_segment, axis=1)

    st.write(f"Added analytical columns. Sample stats:")
    st.write(f"- Products: {df['MON'].nunique():,}")
    st.write(f"- Total listings: {len(df):,}")
    st.write(f"- Price range: {df['Product Prices'].min():,} to {df['Product Prices'].max():,}")

    return df


# =============================================================================
# STEP 3: DISPLAY AND FORMATTING FUNCTIONS
# =============================================================================

def get_product_display_name(mon_value, df):
    """Get display name combining MON and Product Titles"""
    try:
        product_titles = df[df['MON'] == mon_value]['Product Titles'].dropna().unique()
        if len(product_titles) > 0:
            product_name = product_titles[0]
            seller_count = len(df[df['MON'] == mon_value]['Seller'].unique())
            return f"{mon_value} - {product_name} ({seller_count} sellers)"
        else:
            return f"{mon_value} (No title available)"
    except Exception as e:
        st.error(f"Error getting display name for {mon_value}: {e}")
        return mon_value


def format_number(number):
    """Format numbers with comma separators"""
    try:
        return f"{int(number):,}"
    except (ValueError, TypeError):
        return str(number)


def format_price(price):
    """Format price with comma separators and currency"""
    try:
        return f"{int(price):,} تومان"
    except (ValueError, TypeError):
        return f"{price} تومان"


def format_price_range(min_price, max_price):
    """Format price range with comma separators"""
    try:
        return f"{format_price(min_price)} - {format_price(max_price)}"
    except (ValueError, TypeError):
        return "تومان 0"


# =============================================================================
# STEP 4: ANALYTICAL FUNCTIONS
# =============================================================================

def detect_outliers_by_seller(product_data):
    """Detect outliers for each seller using IQR method"""
    sellers = product_data['Seller'].unique()
    outlier_info = {}

    for seller in sellers:
        seller_prices = product_data[product_data['Seller'] == seller]['Product Prices']

        if len(seller_prices) < 2:
            continue

        Q1 = seller_prices.quantile(0.25)
        Q3 = seller_prices.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = seller_prices[(seller_prices < lower_bound) | (seller_prices > upper_bound)]

        if not outliers.empty:
            outlier_info[seller] = {
                'outlier_prices': outliers.tolist(),
                'count': len(outliers),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_percentage': (len(outliers) / len(seller_prices)) * 100
            }

    return outlier_info


def create_insight_cards(product_data, selected_product_display):
    """Create insight cards with key findings"""
    if product_data.empty:
        return

    insights = []

    # Basic stats
    total_sellers = product_data['Seller'].nunique()
    price_range = product_data['Product Prices'].max() - product_data['Product Prices'].min()
    price_range_pct = (price_range / product_data['Product Prices'].min()) * 100

    # Seller count insight
    if total_sellers > 10:
        insights.append(f"🏪 **Wide seller choice**: {total_sellers} different sellers")

    # Price range insight
    if price_range_pct > 50:
        insights.append(f"💰 **High price variation**: {price_range_pct:.1f}% difference between min and max")

    # Cheapest seller insight
    cheapest_row = product_data.loc[product_data['Product Prices'].idxmin()]
    cheapest_seller = cheapest_row['Seller']
    cheapest_platform = cheapest_row['Platform']
    cheapest_price = product_data['Product Prices'].min()
    avg_price = product_data['Product Prices'].mean()
    savings = avg_price - cheapest_price
    savings_pct = (savings / avg_price) * 100

    if savings_pct > 10:
        insights.append(
            f"🎯 **Best deal**: {cheapest_seller} on {cheapest_platform} offers {savings_pct:.1f}% below average")

    # Platform price variation insight
    platform_stats = product_data.groupby('Platform')['Product Prices'].mean()
    if len(platform_stats) > 1:
        platform_variation = (platform_stats.max() - platform_stats.min()) / platform_stats.min() * 100
        if platform_variation > 20:
            insights.append(f"🔄 **Platform price variation**: {platform_variation:.1f}% between marketplaces")

    # Outlier insight
    outlier_info = detect_outliers_by_seller(product_data)
    if outlier_info:
        outlier_count = sum(info['count'] for info in outlier_info.values())
        insights.append(f"🚨 **{outlier_count} outlier prices** detected across sellers")

    # Price consistency insight
    cv = (product_data['Product Prices'].std() / product_data['Product Prices'].mean()) * 100
    if cv < 10:
        insights.append(f"📊 **High price consistency**: Only {cv:.1f}% variation")

    # Display insights
    if insights:
        st.subheader("📈 Key Insights")
        for insight in insights:
            st.info(insight)


# =============================================================================
# STEP 5: CHART CREATION FUNCTIONS
# =============================================================================

def create_main_chart(product_data, selected_product_display, chart_type):
    """Create main chart based on selected type"""
    if product_data.empty:
        return create_empty_figure("No data available for selected product")

    st.write(f"Creating {chart_type} chart for {selected_product_display} with {len(product_data):,} records")

    if chart_type == 'distribution':
        fig = px.histogram(
            product_data,
            x='Product Prices',
            color='Platform',
            title=f'📊 Price Distribution - {selected_product_display}',
            nbins=15,
            opacity=0.8,
            hover_data=['Seller', 'price_segment', 'Platform']
        )
        fig.update_layout(
            xaxis_title='Price (تومان)',
            yaxis_title='Frequency',
            template=CONFIG['chart_templates']
        )

    elif chart_type == 'seller':
        seller_avg = product_data.groupby(['Seller', 'Platform'])['Product Prices'].mean().reset_index()
        fig = px.bar(
            seller_avg,
            x='Seller',
            y='Product Prices',
            color='Platform',
            title=f'🏪 Average Prices by Seller - {selected_product_display}',
            hover_data=['Platform']
        )
        fig.update_layout(
            xaxis_title='Seller',
            yaxis_title='Average Price (تومان)',
            xaxis_tickangle=-45,
            template=CONFIG['chart_templates']
        )

    elif chart_type == 'range':
        fig = go.Figure()
        platforms = product_data['Platform'].unique()
        colors = px.colors.qualitative.Set3

        for i, platform in enumerate(platforms):
            platform_data = product_data[product_data['Platform'] == platform]
            fig.add_trace(go.Scatter(
                x=platform_data['Seller'],
                y=platform_data['Product Prices'],
                mode='markers',
                marker=dict(size=12, color=colors[i % len(colors)], opacity=0.7),
                name=platform,
                hovertemplate='<b>%{x}</b><br>Platform: ' + platform + '<br>Price: %{y:,} تومان<extra></extra>'
            ))

        avg_price = product_data['Product Prices'].mean()
        fig.add_hline(y=avg_price, line_dash="dash", line_color=CONFIG['colors']['danger'],
                      annotation_text=f"Average: {format_price(avg_price)}")

        fig.update_layout(
            title=f'📈 Price Range by Seller - {selected_product_display}',
            xaxis_title='Seller',
            yaxis_title='Price (تومان)',
            xaxis_tickangle=-45,
            template=CONFIG['chart_templates'],
            showlegend=True
        )

    elif chart_type == 'savings':
        product_data = product_data.copy()
        product_data['savings_vs_avg'] = product_data['mean_price'] - product_data['Product Prices']

        fig = px.bar(
            product_data,
            x='Seller',
            y='savings_vs_avg',
            color='Platform',
            title=f'💰 Potential Savings vs Average - {selected_product_display}',
            hover_data=['Product Prices', 'mean_price', 'Platform']
        )
        fig.update_layout(
            xaxis_title='Seller',
            yaxis_title='Savings vs Average (تومان)',
            xaxis_tickangle=-45,
            template=CONFIG['chart_templates']
        )

    elif chart_type == 'segments':
        segment_platform_counts = product_data.groupby(['price_segment', 'Platform']).size().reset_index(name='count')
        fig = px.bar(
            segment_platform_counts,
            x='price_segment',
            y='count',
            color='Platform',
            title=f'📊 Price Segments Distribution - {selected_product_display}',
            barmode='group'
        )
        fig.update_layout(
            xaxis_title='Price Segment',
            yaxis_title='Count',
            template=CONFIG['chart_templates']
        )

    elif chart_type == 'platform':
        fig = px.box(
            product_data,
            x='Platform',
            y='Product Prices',
            title=f'🏪 Platform Price Comparison - {selected_product_display}',
            color='Platform',
            points='all',
            hover_data=['Seller', 'Product Prices']
        )
        fig.update_layout(
            xaxis_title='Platform',
            yaxis_title='Price (تومان)',
            template=CONFIG['chart_templates'],
            showlegend=False
        )

    else:  # Default box plot
        fig = px.box(
            product_data,
            y='Product Prices',
            color='Platform',
            title=f'📦 Price Distribution (Box Plot) - {selected_product_display}',
            hover_data=['Seller', 'Platform']
        )
        fig.update_layout(
            yaxis_title='Price (تومان)',
            template=CONFIG['chart_templates']
        )

    return fig


def create_enhanced_box_plot(product_data, selected_product_display):
    """Create professional box plot with platform separation"""
    fig = px.box(
        product_data,
        x='Seller',
        y='Product Prices',
        color='Platform',
        title=f'<b>Professional Price Analysis - {selected_product_display}</b>',
        points='all',
        hover_data=['Product Prices', 'price_segment', 'Platform'],
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(
        yaxis_title='<b>Price (تومان)</b>',
        xaxis_title='<b>Seller</b>',
        template='plotly_white',
        xaxis_tickangle=-45,
        showlegend=True,
        font=dict(family="Arial, sans-serif", size=12),
        title_font=dict(size=18, color='#2c3e50', family="Arial, sans-serif"),
        plot_bgcolor='rgba(248,249,250,1)',
        paper_bgcolor='white',
        margin=dict(t=80, b=80, l=80, r=40),
        height=500
    )

    if not product_data.empty:
        mean_price = product_data['Product Prices'].mean()
        fig.add_hline(
            y=mean_price,
            line_dash="dot",
            line_color="#3498db",
            line_width=2,
            opacity=0.8,
            annotation_text=f"Mean: {format_price(mean_price)}",
            annotation_position="right",
            annotation_font_size=10,
            annotation_bgcolor="rgba(52, 152, 219, 0.2)"
        )

    return fig


def create_seller_chart(product_data):
    """Create seller comparison chart with platform integration"""
    if product_data.empty:
        return create_empty_figure("No seller data available")

    seller_stats = product_data.groupby(['Seller', 'Platform']).agg({
        'Product Prices': ['mean', 'min', 'max', 'count']
    }).round(2)

    seller_stats.columns = ['Average', 'Minimum', 'Maximum', 'Count']
    seller_stats = seller_stats.reset_index()

    fig = go.Figure()

    platforms = seller_stats['Platform'].unique()
    colors = px.colors.qualitative.Set3

    for i, platform in enumerate(platforms):
        platform_data = seller_stats[seller_stats['Platform'] == platform]
        fig.add_trace(go.Bar(
            name=platform,
            x=platform_data['Seller'],
            y=platform_data['Average'],
            marker_color=colors[i % len(colors)],
            hovertemplate='<b>%{x}</b><br>Platform: ' + platform + '<br>Average: %{y:,} تومان<extra></extra>'
        ))

    fig.update_layout(
        title='🏪 Seller Price Comparison with Platform',
        xaxis_title='Seller',
        yaxis_title='Price (تومان)',
        xaxis_tickangle=-45,
        template=CONFIG['chart_templates'],
        hovermode='x unified'
    )

    return fig


def create_stats_chart(product_data):
    """Create statistics overview chart with platform comparison"""
    if product_data.empty:
        return create_empty_figure("No data available for statistics")

    platform_stats = product_data.groupby('Platform')['Product Prices'].describe().reset_index()

    stats_to_plot = platform_stats.melt(id_vars=['Platform'],
                                        value_vars=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
                                        var_name='Statistic',
                                        value_name='Value')

    fig = px.bar(
        stats_to_plot,
        x='Statistic',
        y='Value',
        color='Platform',
        title='📈 Price Statistics by Platform',
        barmode='group',
        text='Value'
    )

    fig.update_traces(texttemplate='%{y:,}', textposition='outside')
    fig.update_layout(
        xaxis_title='Statistics',
        yaxis_title='Value (تومان)',
        template=CONFIG['chart_templates'],
        showlegend=True
    )
    return fig


# =============================================================================
# STEP 6: DATA TABLE AND KPI FUNCTIONS
# =============================================================================

def create_data_table(product_data):
    """Create the data table component with platform column"""
    if product_data.empty:
        st.write("No data available for selected product")
        return

    table_data = product_data[['Seller', 'Platform', 'Product Prices', 'price_segment', 'price_ratio']].sort_values(
        'Product Prices')

    table_data_formatted = table_data.copy()
    table_data_formatted['Product Prices'] = table_data_formatted['Product Prices'].apply(
        lambda x: f"{x:,.0f} تومان" if pd.notnull(x) else "N/A"
    )
    table_data_formatted['price_ratio'] = table_data_formatted['price_ratio'].apply(
        lambda x: f"{x:+.1f}%" if pd.notnull(x) else "N/A"
    )

    st.dataframe(
        table_data_formatted,
        use_container_width=True,
        height=400
    )


def create_platform_kpi_columns(product_data):
    """Create KPI columns for each platform"""
    if product_data.empty:
        return

    col1, col2 = st.columns(2)

    with col1:
        display_platform_kpis("SnappPay", product_data, CONFIG['colors']['primary'])

    with col2:
        display_platform_kpis("Torob", product_data, CONFIG['colors']['success'])


def display_platform_kpis(platform_name, product_data, color):
    """Display KPIs for a specific platform"""
    platform_data = product_data[product_data['Platform'] == platform_name]

    if platform_data.empty:
        total_products = "0"
        total_sellers = "0"
        price_range = "تومان 0"
        avg_price = "تومان 0"
        max_savings = "تومان 0"
        price_variation = "0%"
        best_seller = "-"
    else:
        total_products = format_number(len(platform_data['MON'].unique()))
        total_sellers = format_number(platform_data['Seller'].nunique())
        min_price = platform_data['Product Prices'].min()
        max_price = platform_data['Product Prices'].max()
        avg_price = platform_data['Product Prices'].mean()

        price_range = format_price_range(min_price, max_price)
        avg_price_str = format_price(avg_price)
        max_savings = format_price(max_price - min_price)
        price_variation = f"{(platform_data['Product Prices'].std() / avg_price * 100):.1f}%" if avg_price > 0 else "0%"
        best_seller = platform_data.loc[
            platform_data['Product Prices'].idxmin(), 'Seller'] if not platform_data.empty else "-"

    st.subheader(f"🏪 {platform_name} Platform Metrics")

    kpi_col1, kpi_col2 = st.columns(2)

    with kpi_col1:
        st.metric("Total Products", total_products)
        st.metric("Total Sellers", total_sellers)
        st.metric("Price Range", price_range)

    with kpi_col2:
        st.metric("Average Price", avg_price_str)
        st.metric("Maximum Savings", max_savings)
        st.metric("Price Variation", price_variation)

    st.write(f"**Best Seller**: {best_seller}")


# =============================================================================
# STEP 7: SEARCH FUNCTIONALITY
# =============================================================================

def advanced_product_search(search_terms, product_list, df):
    """Advanced search that handles random words with permutation matching"""
    if not search_terms:
        return product_list

    search_terms = [term.strip().lower() for term in search_terms.split() if term.strip()]

    if not search_terms:
        return product_list

    matching_products = []

    for product in product_list:
        product_lower = product.lower()
        product_name = df[df['MON'] == product]['Product Titles'].iloc[0] if not df[df['MON'] == product].empty else ""
        product_name_lower = product_name.lower()

        # Search in both MON and Product Titles
        if (all(term in product_lower for term in search_terms) or
                all(term in product_name_lower for term in search_terms)):
            matching_products.append(product)

    return matching_products


def create_empty_figure(message):
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16), xref="paper", yref="paper")
    fig.update_layout(template=CONFIG['chart_templates'])
    return fig


# =============================================================================
# STEP 8: MAIN STREAMLIT APPLICATION
# =============================================================================

def main():
    # Load data
    st.sidebar.title("Data Configuration")
    if st.sidebar.button("Reload Data"):
        st.cache_data.clear()

    # Load data with caching
    @st.cache_data
    def load_data():
        return load_and_preprocess_data()

    df = load_data()
    first_100k = df.iloc[:500000]
    last_100k = df.iloc[-500000:]
    df = pd.concat([first_100k, last_100k])

    # Header
    st.title("📊 Product Price Analysis Dashboard")
    st.markdown("Comprehensive analysis of product pricing across multiple sellers and platforms")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        search_terms = st.text_input(
            "🔍 Search Products",
            placeholder="Example: xiaomi 256 8 or samsung 128gb phone...",
            help="Search by product name, model, or specifications"
        )

    with col2:
        # Get all products for dropdown
        all_products = sorted(df['MON'].unique())

        if search_terms:
            filtered_products = advanced_product_search(search_terms, all_products, df)
            st.info(f"🔍 Found {len(filtered_products):,} products matching: '{search_terms}'")
        else:
            filtered_products = all_products
            st.info(f"📦 Total {len(filtered_products):,} products available")

        # Create dropdown options
        product_options = [get_product_display_name(product, df) for product in filtered_products]
        product_mapping = {get_product_display_name(product, df): product for product in filtered_products}

        if product_options:
            selected_display = st.selectbox(
                "Select Product:",
                options=product_options,
                index=0
            )
            selected_product = product_mapping[selected_display]
        else:
            st.warning("No products found matching your search criteria")
            selected_product = None

    # Chart type selection
    chart_type = st.selectbox(
        "Chart Type:",
        options=[
            'distribution', 'seller', 'range', 'box',
            'savings', 'segments', 'platform'
        ],
        format_func=lambda x: {
            'distribution': '📊 Price Distribution',
            'seller': '🏪 Seller Comparison',
            'range': '📈 Price Range',
            'box': '📦 Box Plot (with Platform Comparison)',
            'savings': '💰 Savings Opportunities',
            'segments': '📊 Price Segments',
            'platform': '🏪 Platform Comparison'
        }[x],
        index=0
    )

    if selected_product:
        # Get display name for the selected product
        selected_product_display = get_product_display_name(selected_product, df)

        # Filter data for selected product
        product_data = df[df['MON'] == selected_product]

        if not product_data.empty:
            # Insight Cards
            create_insight_cards(product_data, selected_product_display)

            # Platform KPI Columns
            st.subheader("📊 Platform Comparison")
            create_platform_kpi_columns(product_data)

            # Main Chart
            st.subheader("📈 Price Analysis Chart")
            if chart_type == 'box':
                fig_main = create_enhanced_box_plot(product_data, selected_product_display)
            else:
                fig_main = create_main_chart(product_data, selected_product_display, chart_type)

            st.plotly_chart(fig_main, use_container_width=True)

            # Secondary Charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏪 Seller Comparison")
                fig_seller = create_seller_chart(product_data)
                st.plotly_chart(fig_seller, use_container_width=True)

            with col2:
                st.subheader("📊 Statistics Overview")
                fig_stats = create_stats_chart(product_data)
                st.plotly_chart(fig_stats, use_container_width=True)

            # Data Table
            st.subheader("📋 Detailed Price Data")
            create_data_table(product_data)
        else:
            st.error("No data available for the selected product")
    else:
        st.warning("Please select a product to view the analysis")

    # Dataset summary in sidebar
    st.sidebar.subheader("Dataset Summary")
    st.sidebar.write(f"**Total records**: {len(df):,}")
    st.sidebar.write(f"**Unique products**: {df['MON'].nunique():,}")
    st.sidebar.write(f"**Unique sellers**: {df['Seller'].nunique():,}")
    st.sidebar.write(f"**Platforms**: {', '.join(df['Platform'].unique())}")
    st.sidebar.write(f"**Price range**: {df['Product Prices'].min():,} to {df['Product Prices'].max():,} تومان")


if __name__ == "__main__":
    main()
